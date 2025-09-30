import json

# flow_dataset = json.load(open("project_default_directory/Sth-Sth/something_something_flow/something-something-v2-train.json"))
# video_dataset = json.load(open("project_default_directory/Sth-Sth/something_something_detections/train_dataset.json"))
#
# # flow_dataset = json.load(open("project_default_directory/Sth-Sth/something_something_detections/val_dataset_flow_aligned.json"))
# # video_dataset = json.load(open("project_default_directory/Sth-Sth/something_something_detections/val_dataset.json"))
# #
# # Extract ids from each list
# ids_list1 = {d["id"] for d in flow_dataset}
# ids_list2 = {d["id"] for d in video_dataset}
#
# # Find common ids
# common_ids = ids_list1.intersection(ids_list2)
#
# # Filter dictionaries based on common ids and keep them in separate lists
# common_elements_list1 = [d for d in flow_dataset if d["id"] in common_ids]
# common_elements_list2 = [d for d in video_dataset if d["id"] in common_ids]
#
# print("Common elements in list1:", len(common_elements_list1))
# print("Common elements in list2:", len(common_elements_list2))
#
# json.dump(common_elements_list1, open(
#     "project_default_directory/Sth-Sth/something_something_detections/kkontras_flow_train.json",
#     "w"))
# json.dump(common_elements_list2, open(
#     "project_default_directory/Sth-Sth/something_something_detections/kkontras_video_train.json",
#     "w"))

# json.dump(common_elements_list1, open(
#     "project_default_directory/Sth-Sth/something_something_detections/kkontras_flow_val.json",
#     "w"))
# json.dump(common_elements_list2, open(
#     "project_default_directory/Sth-Sth/something_something_detections/kkontras_video_val.json",
#     "w"))

flow_dataset = json.load(open("project_default_directory/Sth-Sth/something_something_detections/kkontras_flow_train.json"))
video_dataset = json.load(open("project_default_directory/Sth-Sth/something_something_detections/kkontras_video_train.json"))
#
# flow_dataset = json.load(open("project_default_directory/Sth-Sth/something_something_detections/kkontras_flow_val.json"))
# video_dataset = json.load(open("project_default_directory/Sth-Sth/something_something_detections/kkontras_video_val.json"))
#
flow_resource_path = "project_default_directory/Sth-Sth/flow_dataset.hdf5"
video_resource_path = "project_default_directory/Sth-Sth/dataset.hdf5"

import h5py

flow_resource = h5py.File(flow_resource_path, "r", libver="latest", swmr=True)
video_resource = h5py.File(video_resource_path, "r", libver="latest", swmr=True)

count = 0
for i in range(len(flow_dataset)):
    if flow_dataset[i]["id"] != video_dataset[i]["id"]:
        count += 1
        print("Problem {}".format(i))
    else:
        if len(flow_resource["{}".format(flow_dataset[i]["id"])]) + 1 != len(video_resource["{}".format(video_dataset[i]["id"])]):
            print(
                "Difference in {}-{} and {}-{}".format(
                    len(flow_resource["{}".format(flow_dataset[i]["id"])]) + 1,
                    len(video_resource["{}".format(video_dataset[i]["id"])]),
                    i,
                    flow_dataset[i]["id"],
                )
            )
print(count)
