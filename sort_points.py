import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment

# file_path_r = 'map_v2_r_xyz.csv'
# file_path_l = 'map_v2_l_xyz.csv'
# file_path_al = 'map_v2_al_xyz.csv'
# file_path_ar = 'map_v2_ar_xyz.csv'

file_path_lane_1 = '1xyz.csv'
file_path_lane_2 = '2xyz.csv'
file_path_lane_3 = '3xyz.csv'

columns_to_read = ['x', 'y', 'z']

# r_map = pd.read_csv(file_path_r, usecols=columns_to_read, encoding='utf-8')
# l_map = pd.read_csv(file_path_l, usecols=columns_to_read, encoding='utf-8')
# ar_map = pd.read_csv(file_path_ar, usecols=columns_to_read, encoding='utf-8')
# al_map = pd.read_csv(file_path_al, usecols=columns_to_read, encoding='utf-8')

lane_1 = pd.read_csv(file_path_lane_1, usecols=columns_to_read, encoding='utf-8')
lane_2 = pd.read_csv(file_path_lane_2, usecols=columns_to_read, encoding='utf-8')
lane_3 = pd.read_csv(file_path_lane_3, usecols=columns_to_read, encoding='utf-8')


# r_map_array = r_map.values
# l_map_array = l_map.values
# ar_map_array = ar_map.values
# al_map_array = al_map.values

lane_1_array = lane_1.values
lane_2_array = lane_2.values
lane_3_array = lane_3.values

SAVECSV = False
save_file_name = 'lane_3.csv'
points_1 = lane_1_array
points_2 = lane_2_array
points_3 = lane_3_array

dist_matrix_1 = distance_matrix(points_1, points_1)
dist_matrix_2 = distance_matrix(points_2, points_2)
dist_matrix_3 = distance_matrix(points_3, points_3)

######################################### previous offsets #########################################
## zoker_l_1m : ccw // start point ; 1250 // outlier ; 175`184
## zoker_l_5m : ccw // start point ; 1250 // outlier ; 35`36
## zoker_r_5m : cw // start point ; 1000 // outlier ; 137
## zoker_r_1m : cw // start point ; 1000 // outlier ; 681`688

# map_v2_l_1m : ccw // start point ; 1400 // outlier ; (delete following order) 359:370 -> 727:773
# map_v2_l_5m : ccw // start point ; 1400 // outlier ; 1--
# map_v2_r_1m : ccw // start point ; 1400 // outlier ; (delete following order) 3363:3371 -> 3369:3372 ->439:442 -> 1530:1533 -> 2294:2297
# map_v2_r_5m : ccw // start point ; 1400 // outlier ; 673
######################################### previous offsets #########################################

# 한붓그리기 순서대로 정렬
def order_points_knn(points, dist_matrix):
    num_points = len(points)
    # current_point_idx = np.argmin(points[:,0])
    current_point_idx = 100
    ordered_points = [points[current_point_idx]]
    remaining_points = set(range(1, num_points))

    while remaining_points:
        distances = dist_matrix[current_point_idx, list(remaining_points)]
        nearest_point_idx = list(remaining_points)[np.argmin(distances)]
        ordered_points.append(points[nearest_point_idx])
        current_point_idx = nearest_point_idx
        remaining_points.remove(nearest_point_idx)

    return np.array(ordered_points)




def resample_points(points, interval):
    resampled_points = [points[0]]
    accumulated_distance = 0.0

    for i in range(1, len(points)):
        start_point = points[i-1]
        end_point = points[i]
        segment_distance = np.linalg.norm(end_point - start_point)

        while accumulated_distance + segment_distance >= interval:
            t = (interval - accumulated_distance) / segment_distance
            new_point = start_point + t * (end_point - start_point)
            resampled_points.append(new_point)
            start_point = new_point
            segment_distance = np.linalg.norm(end_point - start_point)
            accumulated_distance = 0.0

        accumulated_distance += segment_distance

    return np.array(resampled_points)

interval = 1.0
weight = int(5.0/interval)

sorted_points_1 = order_points_knn(points_1, dist_matrix_1)
resampled_points_1 = resample_points(sorted_points_1, interval)
sorted_points_2 = order_points_knn(points_2, dist_matrix_2)
resampled_points_2 = resample_points(sorted_points_2, interval)
resampled_points_2 = resampled_points_2[::-1]                       ## order reverse to ccw order
sorted_points_3 = order_points_knn(points_3, dist_matrix_3)
resampled_points_3 = resample_points(sorted_points_3, interval)
# resampled_points = resample_points(sorted_points)[::-1] ## order reverse
# resampled_points = resampled_points[::-1] ## order reverse



######################################### removing outliers #########################################
# for i in range(11):
#     resampled_points = np.delete(resampled_points, 359, axis=0)
# for i in range(46):
#     resampled_points = np.delete(resampled_points, 727, axis=0)
# for i in range(3):
#     resampled_points = np.delete(resampled_points, 439, axis=0)
# for i in range(3):
#     resampled_points = np.delete(resampled_points, 1530, axis=0)
# for i in range(3):
#     resampled_points = np.delete(resampled_points, 2294, axis=0)
######################################### removing outliers #########################################



print(len(resampled_points_1))          
print(len(resampled_points_2))
print(len(resampled_points_3))


######################################### visualize #########################################

# plt.scatter(resampled_points[727:773 ,0], resampled_points[727:773, 1], c='cyan',s = 80,  label='Data points')
plt.scatter(resampled_points_1[0*weight:300*weight ,0], resampled_points_1[0*weight:300*weight, 1], c='green',s = 0.8,  label='Data points')
plt.scatter(resampled_points_1[300*weight:500*weight ,0], resampled_points_1[300*weight:500*weight, 1], c='red',s = 0.8,  label='Data points')
plt.scatter(resampled_points_1[500*weight:600*weight ,0], resampled_points_1[500*weight:600*weight, 1], c='magenta',s = 0.8,  label='Data points')
plt.scatter(resampled_points_1[600*weight:700*weight ,0], resampled_points_1[600*weight:700*weight, 1], c='yellow',s = 0.8,  label='Data points')
plt.scatter(resampled_points_1[700*weight: ,0], resampled_points_1[700*weight:, 1], c='black',s = 0.8,  label='Data points')
plt.scatter(resampled_points_3[: ,0], resampled_points_3[:, 1], c='black',s = 0.8,  label='Data points')
plt.scatter(resampled_points_2[: ,0], resampled_points_2[:, 1], c='black',s = 0.8,  label='Data points')
# plt.scatter(al_points[: ,0], al_points[:, 1], c='black',s = 10,  label='Data points')
# plt.scatter(ar_points[: ,0], ar_points[:, 1], c='black',s = 10,  label='Data points')
plt.show()
######################################### visualize #########################################

######################################### save sorted file #########################################
if SAVECSV:
    resampled_points = pd.DataFrame(resampled_points_3, columns=['x', 'y', 'z'])
    resampled_points.to_csv(save_file_name, index = False)
    print('csv saved :', save_file_name)
    print(len(resampled_points))


######################################### save sorted file #########################################

