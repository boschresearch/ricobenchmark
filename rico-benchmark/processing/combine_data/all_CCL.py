#copyright (c) 2025 Robert Bosch GmbH, AGPL-3.0
from utils import *

folders = {
    "Woodscape": " woodscape/annotations_day_ccl.json",
    "DENSEgated": " dense/annotations_gated_ccl.json",
    "nuImages": " nuImages/annotations_more_bike_ccl.json",
    "FishEye8K": " fisheye8k/annotations_ccl.json",
    "SHIFT": " shift/annotations_ccl.json",
    "VisDrone": " visdrone/annotations_ccl.json",
    "FLIR": " flir/annotations_more_obj_ccl.json",
    "BDD100K": " bdd100k/annotations_ccl.json",
}

save_path = " multi-ad-ccl"

print("Number of datasets:", len(folders))


data_all = load_and_process_data(folders)

tasks_dicts = calculate_task_sequences(data_all)

splits = {}
for task_name in tasks_dicts:
    print(task_name)
    best_splits, smallest_error, total_items = get_best_split(
        tasks_dicts[task_name], 0.6, 0.1, 1000, 0.0003
    )
    splits[task_name] = best_splits
    print("Smallest error:", smallest_error)
    print("Train ratio:", sum([i[1] for i in best_splits[0]]) / total_items)
    print("Val ratio:", sum([i[1] for i in best_splits[1]]) / total_items)
    print("Test ratio:", sum([i[1] for i in best_splits[2]]) / total_items)
    print()

smallest_training_set, smallest_val_set, smallest_test_set, total, ratios = (
    calculate_smallest_sets(splits)
)

print(smallest_training_set, smallest_val_set, smallest_test_set, total)
print(ratios)

splits_matching = create_splits_matching(
    splits, smallest_training_set, smallest_val_set, smallest_test_set
)

data_all = process_data(data_all)

mean_std = load_data_from_json('mean_std/stats.json')

for task_name in data_all:
    for i in range(len(data_all[task_name])):
        data_all[task_name][i]['mean'] = [v* 255 for v in mean_std['mean_task'][task_name]]
        data_all[task_name][i]['std'] = [v*255 for v in mean_std['std_task'][task_name]]

data_train, data_val, data_test, data_train_db, data_val_db, data_test_db = (
    generate_data_sets(
        splits_matching,
        data_all,
        smallest_training_set,
        smallest_val_set,
        smallest_test_set,
    )
)
data_complete_train = combine_data(data_train)
data_complete_train_db = combine_data(data_train_db)
data_complete_val = combine_data(data_val)
data_complete_val_db = combine_data(data_val_db)
data_complete_test = combine_data(data_test)
data_complete_test_db = combine_data(data_test_db)

print(len(data_complete_train))
print(len(data_complete_val))
print(len(data_complete_test))
print(
    "Size complete:",
    len(data_complete_train) + len(data_complete_val) + len(data_complete_test),
)

# save compelte debug data
save_data_as_json(data_complete_train_db, save_path + "/train_db.json")
save_data_as_json(data_complete_val_db, save_path + "/val_db.json")
save_data_as_json(data_complete_test_db, save_path + "/test_db.json")
print("saved complete debug data")

# save complete data
save_data_as_json(data_complete_train, save_path + "/train.json")
save_data_as_json(data_complete_val, save_path + "/val.json")
save_data_as_json(data_complete_test, save_path + "/test.json")
print("saved complete data")

# save individiual data
for folder in data_train:
    for type, datastorage in zip(
        ["train", "val", "test"], [data_train, data_val, data_test]
    ):
        save_data_as_json(
            datastorage[folder], save_path + "/" + type + "_" + folder + ".json"
        )
    for type, datastorage in zip(
        ["train", "val", "test"], [data_train_db, data_val_db, data_test_db]
    ):
        save_data_as_json(
            datastorage[folder], save_path + "/" + type + "_db_" + folder + ".json"
        )
    print("saved data for", folder)