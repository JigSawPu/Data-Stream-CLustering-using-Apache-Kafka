import numpy as np

def find_second_smallest_index(arr):
    min_num = min(arr[0], arr[1])
    second_min = max(arr[0], arr[1])
    for i in range(2, len(arr)):
        if arr[i] < min_num:
            second_min = min_num
            min_num = arr[i]
        elif arr[i] < second_min and arr[i] != min_num:
            second_min = arr[i]
    return second_min
    
    
def calculate_silhouette(i):
    silhouette_array=np.array([])
    for item in clusters[i]:
        a=((clusters[i]-item)**2).sum(axis=1).mean()
        b_array=np.array([])
        for j in range(num_clusters):
            if j==i:
                continue
            else:
                pre=((clusters[j]-item)**2).sum(axis=1).mean()
                b_array=np.append(b_array, pre)
        b=b_array[np.argmin(b_array)]
        silhouette_score=(b-a)/max(b,a)
        silhouette_array=np.append(silhouette_array, silhouette_score)
    return silhouette_array
