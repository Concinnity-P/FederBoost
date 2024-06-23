import numpy as np
import concurrent.futures
from SFXGBoost.Model import SFXGBoost, devide_D_Train
from SFXGBoost.config import Config, MyLogger
from SFXGBoost.dataset.datasetRetrieval import getDataBase
from ddsketch import DDSketch
from SFXGBoost.view.plotter import plot_loss

dataset = 'healthcare'
NUM_CLIENTS = 2
DATA_DEVISION = [1/NUM_CLIENTS] * NUM_CLIENTS

def setup_config():
    config = Config(experimentName="experiment 1",
                    nameTest=dataset + " test",
                    model="normal",
                    dataset=dataset,
                    lam=0,  # 0.1 10
                    gamma=0,  # 0.5
                    alpha=0.0,
                    learning_rate=0.3,
                    max_depth=8,
                    max_tree=5,
                    nBuckets=100,
                    save=False,
                    data_devision=DATA_DEVISION,
                    train_size=10_000,
                    client=0,
                    num_client=NUM_CLIENTS)
    
    configs = []
    for i in range(config.num_client + 1):
        configs.append(Config(experimentName="experiment 1",
                              nameTest=config.nameTest,
                              model=config.model,
                              dataset=dataset,
                              lam=config.lam,  # 0.1 10
                              gamma=config.gamma,  # 0.5
                              alpha=config.alpha,
                              learning_rate=config.learning_rate,
                              max_depth=config.max_depth,
                              max_tree=config.max_tree,
                              nBuckets=config.nBuckets,
                              save=False,
                              data_devision=config.data_devision,
                              train_size=config.train_size,
                              client=i,  # 0 is server
                              num_client=config.num_client))
    return config, configs

def setup_data(config):
    POSSIBLE_PATHS = ["e:\\oneDrive\\UOB\\Federated_XGBoost_Python-main\\data\\"]
    X_train, y_train, X_test, y_test, fName, X_shadow, y_shadow = getDataBase(config.dataset, POSSIBLE_PATHS, False, config.train_size)()
    
    total_users = config.num_client  # participants
    X_train_list = []
    y_train_list = []

    for rank in range(total_users):
        rank += 1
        X_train_my, y_train_my = devide_D_Train(X_train, y_train, rank, config.data_devision)
        X_train_list.append(X_train_my)
        y_train_list.append(y_train_my)
    
    return X_train_list, y_train_list, fName, total_users

def setup_clients(configs, X_train_list, y_train_list, fName):
    res_to_merge = []
    bst_clients = []
    for i in range(len(X_train_list)):
        bst_client = SFXGBoost(configs[i + 1], MyLogger(configs[i + 1]).logger)
        res = bst_client.participant_fit(X_train_list[i], y_train_list[i], fName)
        res_to_merge.append(res)
        bst_clients.append(bst_client)
    return res_to_merge, bst_clients

def merge_results(res_to_merge, config, fName):
    total_users = config.num_client
    splitCandidates = []
    for i in range(len(res_to_merge[0])):
        if all([isinstance(res_to_merge[p][i][1], np.ndarray) for p in range(total_users)]):
            combined_array = np.concatenate([res_to_merge[p][i][1] for p in range(total_users)], axis=0)
            splitCandidates.append(np.unique(combined_array))
        else:
            sketch = DDSketch()
            for j in range(total_users):
                sketch.merge(res_to_merge[j][i][0])
            quantiles = np.array([sketch.get_quantile_value(q / config.nBuckets) for q in range(0, config.nBuckets, 1)])
            splitCandidates.append(quantiles)
    
    splitCandidates_dict = {fName[i]: splitCandidates[i] for i in range(len(splitCandidates))}
    return splitCandidates_dict, splitCandidates

def update_clients(bst_clients, splitCandidates_dict):
    for i in range(len(bst_clients)):
        bst_clients[i].setquantiles(splitCandidates_dict)

def train_federated_model(config:Config, bst:SFXGBoost, bst_clients:list[SFXGBoost], total_users):
    for t in range(config.max_tree):
        print(f"Tree {t}:")
        for d in range(config.max_depth):
            print(f"Depth {d}:")

            # Parallelize participant_boost calls
            GHs = []
            with concurrent.futures.ProcessPoolExecutor(max_workers=total_users) as executor:
                # futures = [executor.submit(bst_clients[i].participant_boost, t, d) for i in range(total_users)]
                futures = {executor.submit(bst_clients[i].participant_boost, t, d): i for i in range(total_users)}
                for future in concurrent.futures.as_completed(futures):
                    GHs.append(future.result())

            update_info = bst.server_boost(GHs, t, d)
            train_losses = np.array([])
            test_losses = np.array([])

            for i in range(total_users):
                loss = bst_clients[i].participant_update(update_info, t, d)  # return train loss and test loss
                train_losses = np.append(train_losses, loss[0])
                test_losses = np.append(test_losses, loss[1])

            if d == config.max_depth - 1:
                print(f"Train loss: {np.mean(train_losses)}")
                print(f"Test loss: {np.mean(test_losses)}")
                bst.losslog_train.append(np.mean(train_losses))
                bst.losslog_test.append(np.mean(test_losses))
                if t == config.max_tree - 1:
                    plot_loss(bst.losslog_train, bst.losslog_test, config)

def main():
    config, configs = setup_config()
    X_train_list, y_train_list, fName, total_users = setup_data(config)
    res_to_merge, bst_clients = setup_clients(configs, X_train_list, y_train_list, fName)
    splitCandidates_dict, splitCandidates = merge_results(res_to_merge, config, fName)
    update_clients(bst_clients, splitCandidates_dict)
    
    bst = SFXGBoost(configs[0], MyLogger(configs[0]).logger)  # server model
    bst.server_fit(fName, splitCandidates)
    
    train_federated_model(config, bst, bst_clients, total_users)

if __name__ == "__main__":
    main()
