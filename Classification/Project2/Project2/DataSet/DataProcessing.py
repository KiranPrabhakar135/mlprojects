import pandas as pd
import numpy as np
import math
from os import path


def read_data(file):
    data = pd.read_csv(file)
    return data


def get_shuffled_data(pairs, count):
    return pairs.sample(frac=1).head(count)


def process_different_pairs_human_observed_data(pairs):
    different_pairs_size = pairs.shape[0]
    different_blocks_count = math.floor(different_pairs_size/500)
    processed_diff_df = pd.DataFrame()
    for i in range(different_blocks_count):
        k = i * 500
        processed_diff_df = processed_diff_df.append(differentPairsHumanObservedData[k: k + 5])
    return processed_diff_df


def process_same_pairs_gsc_data(pairs):
    pairs_size = pairs.shape[0]
    different_blocks_count = math.floor(pairs_size/50)
    processed_same_df = pd.DataFrame()
    for i in range(different_blocks_count):
        k = i * 50
        processed_same_df = processed_same_df.append(samePairsGscData[k: k + 2])
    return processed_same_df


def process_different_pairs_gsc_data(pairs):
    different_pairs_size = pairs.shape[0]
    different_blocks_count = math.floor(different_pairs_size/500)
    processed_diff_df = pd.DataFrame()
    for i in range(different_blocks_count):
        k = i * 500
        processed_diff_df = processed_diff_df.append(differentPairsGscData[k: k + 2])
    return processed_diff_df


def get_concatenated_and_subtracted_features(data, human_or_gsc_data, is_human_data):
    result = pd.merge(left=data, right=human_or_gsc_data, how='inner', left_on=['img_id_A'], right_on= ['img_id'])
    result = pd.merge(left=result, right=human_or_gsc_data, how='inner', left_on=['img_id_B'], right_on=['img_id'])

    if is_human_data:
        imgA_features = result.loc[:,'f1_x':'f9_x']
        imgB_features = result.loc[:,'f1_y':'f9_y']
    else:
        imgA_features = result.loc[:, 'f1_x':'f512_x']
        imgB_features = result.loc[:, 'f1_y':'f512_y']

    concatenated_array = imgA_features.join(imgB_features)
    subtracted_array = np.absolute(np.subtract(imgA_features, imgB_features))

    return concatenated_array, subtracted_array


def generate_data(same_pairs_data, different_pairs_data):
    total_data = pd.concat([same_pairs_data, different_pairs_data])
    total_data = remove_irrelevant_features(total_data)
    total_data = total_data.sample(frac=1)
    total_data_count = total_data.shape[0]
    eighty_percent_data = math.ceil(total_data_count * 0.8)
    ninety_percent_data = eighty_percent_data + 1 + math.floor(total_data_count * 0.1)
    training_data = total_data.head(eighty_percent_data)
    validation_data = total_data.iloc[eighty_percent_data+1:ninety_percent_data, :]
    testing_data = total_data.iloc[ninety_percent_data:, :]
    return training_data,validation_data,testing_data,total_data


def remove_irrelevant_features(data):
    data_numeric = data
    for key,value in data_numeric.iteritems():
        variance = np.sum(np.absolute(np.array(value)))
        if variance == 0:
            data_numeric.drop(key, axis=1, inplace=True)
    return data_numeric


print("---------------Data Pre Processing Started--------------")
# Human Observed Data
humanObservedData = pd.DataFrame(read_data(path.realpath("DataSet/HumanObserved/HumanObserved-Features-Data.csv")))
samePairsHumanObservedData = pd.DataFrame(read_data(path.realpath("DataSet/HumanObserved/same_pairs.csv")))
differentPairsHumanObservedData = pd.DataFrame(read_data(path.realpath("DataSet/HumanObserved/diffn_pairs.csv")))
processedDifferentPairsHumanObservedData = get_shuffled_data(differentPairsHumanObservedData, samePairsHumanObservedData.shape[0])


same_paris_human_data = get_concatenated_and_subtracted_features(samePairsHumanObservedData, humanObservedData, True)
concatenated_same_paris_human_data = same_paris_human_data[0]
concatenated_same_paris_human_target = pd.DataFrame(np.ones((concatenated_same_paris_human_data.shape[0], 1), dtype=int))
concatenated_same_paris_human_data = concatenated_same_paris_human_data.join(concatenated_same_paris_human_target)
subtracted_same_pairs_human_data = same_paris_human_data[1]
subtracted_same_paris_human_target = pd.DataFrame(np.ones((subtracted_same_pairs_human_data.shape[0], 1), dtype=int))
subtracted_same_pairs_human_data = subtracted_same_pairs_human_data.join(subtracted_same_paris_human_target)


different_pairs_human_data = get_concatenated_and_subtracted_features(processedDifferentPairsHumanObservedData, humanObservedData, True)
concatenated_different_paris_human_data = different_pairs_human_data[0]
concatenated_different_paris_human_target = pd.DataFrame(np.zeros((concatenated_different_paris_human_data.shape[0], 1), dtype=int))
concatenated_different_paris_human_data = concatenated_different_paris_human_data.join(concatenated_different_paris_human_target)
subtracted_different_pairs_human_data = different_pairs_human_data[1]
subtracted_different_paris_human_target = pd.DataFrame(np.zeros((subtracted_different_pairs_human_data.shape[0], 1), dtype=int))
subtracted_different_pairs_human_data = subtracted_different_pairs_human_data.join(subtracted_different_paris_human_target)


total_concatenated_human_generated_data = generate_data(concatenated_same_paris_human_data, concatenated_different_paris_human_data)
concatenated_human_training_data = total_concatenated_human_generated_data[0].iloc[:, :-1]
concatenated_human_training_target_data = total_concatenated_human_generated_data[0].iloc[:,-1]
concatenated_human_validation_data = total_concatenated_human_generated_data[1].iloc[:, :-1]
concatenated_human_validation_target_data = total_concatenated_human_generated_data[1].iloc[:,-1]
concatenated_human_testing_data = total_concatenated_human_generated_data[2].iloc[:, :-1]
concatenated_human_testing_target_data = total_concatenated_human_generated_data[2].iloc[:,-1]
total_concatenated_human_data = total_concatenated_human_generated_data[3].iloc[:, :-1]
total_concatenated_human_target_data = total_concatenated_human_generated_data[3].iloc[:, -1]

total_subtracted_human_generated_data = generate_data(subtracted_same_pairs_human_data, subtracted_different_pairs_human_data)
subtracted_human_training_data = total_subtracted_human_generated_data[0].iloc[:, :-1]
subtracted_human_training_target_data = total_subtracted_human_generated_data[0].iloc[:, -1]
subtracted_human_validation_data = total_subtracted_human_generated_data[1].iloc[:, :-1]
subtracted_human_validation_target_data = total_subtracted_human_generated_data[1].iloc[:, -1]
subtracted_human_testing_data = total_subtracted_human_generated_data[2].iloc[:, :-1]
subtracted_human_testing_target_data = total_subtracted_human_generated_data[2].iloc[:, -1]
total_subtracted_human_data = total_subtracted_human_generated_data[3].iloc[:, :-1]
total_subtracted_human_target_data = total_subtracted_human_generated_data[3].iloc[:, -1]

# GSC data
gscObservedData = pd.DataFrame(read_data(path.realpath("DataSet/GSC/GSC-Features.csv")))
samePairsGscData = pd.DataFrame(read_data(path.realpath("DataSet/GSC/same_pairs.csv")))
differentPairsGscData = pd.DataFrame(read_data(path.realpath("DataSet/GSC/diffn_pairs.csv")))
processedSamePairsGscData = get_shuffled_data(samePairsGscData, 3000)
processedDifferentPairsGscData = get_shuffled_data(differentPairsGscData, 3000)


same_paris_gsc_data = get_concatenated_and_subtracted_features(processedSamePairsGscData, gscObservedData, False)
concatenated_same_paris_gsc_data = same_paris_gsc_data[0]
concatenated_same_paris_gsc_target = pd.DataFrame(np.ones((concatenated_same_paris_gsc_data.shape[0], 1), dtype=int))
concatenated_same_paris_gsc_data = concatenated_same_paris_gsc_data.join(concatenated_same_paris_gsc_target)
subtracted_same_pairs_gsc_data = same_paris_gsc_data[1]
subtracted_same_paris_gsc_target = pd.DataFrame(np.ones((subtracted_same_pairs_gsc_data.shape[0], 1), dtype=int))
subtracted_same_pairs_gsc_data = subtracted_same_pairs_gsc_data.join(subtracted_same_paris_gsc_target)


different_pairs_gsc_data = get_concatenated_and_subtracted_features(processedDifferentPairsGscData, gscObservedData, False)
concatenated_different_paris_gsc_data = different_pairs_gsc_data[0]
concatenated_different_paris_gsc_target = pd.DataFrame(np.zeros((concatenated_different_paris_gsc_data.shape[0], 1), dtype=int))
concatenated_different_paris_gsc_data = concatenated_different_paris_gsc_data.join(concatenated_different_paris_gsc_target)
subtracted_different_pairs_gsc_data = different_pairs_gsc_data[1]
subtracted_different_paris_gsc_target = pd.DataFrame(np.zeros((subtracted_different_pairs_gsc_data.shape[0], 1), dtype=int))
subtracted_different_pairs_gsc_data = subtracted_different_pairs_gsc_data.join(subtracted_different_paris_gsc_target)


total_concatenated_gsc_generated_data = generate_data(concatenated_same_paris_gsc_data, concatenated_different_paris_gsc_data)
concatenated_gsc_training_data = total_concatenated_gsc_generated_data[0].iloc[:, :-1]
concatenated_gsc_training_target_data = total_concatenated_gsc_generated_data[0].iloc[:, -1]
concatenated_gsc_validation_data = total_concatenated_gsc_generated_data[1].iloc[:, :-1]
concatenated_gsc_validation_target_data = total_concatenated_gsc_generated_data[1].iloc[:, -1]
concatenated_gsc_testing_data = total_concatenated_gsc_generated_data[2].iloc[:, :-1]
concatenated_gsc_testing_target_data = total_concatenated_gsc_generated_data[2].iloc[:, -1]
total_concatenated_gsc_data = total_concatenated_gsc_generated_data[3].iloc[:, :-1]
total_concatenated_gsc_target_data = total_concatenated_gsc_generated_data[3].iloc[:, -1]

total_subtracted_gsc_generated_data = generate_data(subtracted_same_pairs_gsc_data, subtracted_different_pairs_gsc_data)
subtracted_gsc_training_data = total_subtracted_gsc_generated_data[0].iloc[:, :-1]
subtracted_gsc_training_target_data = total_subtracted_gsc_generated_data[0].iloc[:, -1]
subtracted_gsc_validation_data = total_subtracted_gsc_generated_data[1].iloc[:, :-1]
subtracted_gsc_validation_target_data = total_subtracted_gsc_generated_data[1].iloc[:, -1]
subtracted_gsc_testing_data = total_subtracted_gsc_generated_data[2].iloc[:, :-1]
subtracted_gsc_testing_target_data = total_subtracted_gsc_generated_data[2].iloc[:, -1]
total_subtracted_gsc_data = total_subtracted_gsc_generated_data[3].iloc[:, :-1]
total_subtracted_gsc_target_data = total_subtracted_gsc_generated_data[3].iloc[:, -1]


print("----------------Human data shapes--------------")
print("Concatenated Human Training data shape: " + str(concatenated_human_training_data.shape))
print("Concatenated Human Validation data shape: " + str(concatenated_human_validation_data.shape))
print("Concatenated Human Testing data shape: " + str(concatenated_human_testing_data.shape))
print("Subtracted Human Training data shape: " + str(subtracted_human_training_data.shape))
print("Subtracted Human Validation data shape: " + str(subtracted_human_validation_data.shape))
print("Subtracted Human Testing data shape: " + str(subtracted_human_testing_data.shape))
print("----------------Human target data shapes------------")
print("Concatenated Human Training target data shape: " + str(concatenated_human_training_target_data.shape))
print("Concatenated Human Validation target data shape: " + str(concatenated_human_validation_target_data.shape))
print("Concatenated Human Testing target data shape: " + str(concatenated_human_testing_target_data.shape))
print("Subtracted Human Training target data shape: " + str(subtracted_human_training_target_data.shape))
print("Subtracted Human Validation target data shape: " + str(subtracted_human_validation_target_data.shape))
print("Subtracted Human Testing target data shape: " + str(subtracted_human_testing_target_data.shape))


print("----------------GSC data shapes--------------------")
print("Concatenated GSC Training data shape: " + str(concatenated_gsc_training_data.shape))
print("Concatenated GSC Validation data shape: " + str(concatenated_gsc_validation_data.shape))
print("Concatenated GSC Testing data shape: " + str(concatenated_gsc_testing_data.shape))
print("Subtracted GSC Training data shape: " + str(subtracted_gsc_training_data.shape))
print("Subtracted GSC Validation data shape: " + str(subtracted_gsc_validation_data.shape))
print("Subtracted GSC Training data shape: " + str(subtracted_gsc_testing_data.shape))
print("---------------GSC target data shapes--------------")
print("Concatenated GSC Training target data shape: " + str(concatenated_gsc_training_target_data.shape))
print("Concatenated GSC Validation target data shape: " + str(concatenated_gsc_validation_target_data.shape))
print("Concatenated GSC Testing target data shape: " + str(concatenated_gsc_testing_target_data.shape))
print("Subtracted GSC Training target data shape: " + str(subtracted_gsc_training_target_data.shape))
print("Subtracted GSC Validation target data shape: " + str(subtracted_gsc_validation_target_data.shape))
print("Subtracted GSC Testing target data shape: " + str(subtracted_gsc_testing_target_data.shape))
print("---------------Data Pre Processing Completed--------------")