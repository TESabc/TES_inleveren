from dataloader.kkg_json_loader import KadasterJsonLoader

ref1 = r'..\Training_data\Main_Training_data\training_data_brt_filter.json'
ref2 = r'..\Training_data\Train_dev_split_for_hyperparameter_tuning\dev_data_brt_filter.json'
ref3 = r'..\Training_data\Train_dev_split_for_hyperparameter_tuning\train_data_brt_filter.json'

data_loader1 = KadasterJsonLoader(ref1)
data_loader2 = KadasterJsonLoader(ref2)
data_loader3 = KadasterJsonLoader(ref3)

print(data_loader1.get_question_by_idx(data_loader1.get_idx_by_question_id(1)))
print(data_loader2.get_question_by_idx(data_loader1.get_idx_by_question_id(1)))
print(data_loader3.get_question_by_idx(data_loader1.get_idx_by_question_id(1)))



