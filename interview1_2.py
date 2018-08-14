import numpy as np

def look_around(x,y,input_image, mark_image):
    stack_results = []
    N = input_image.shape[0]
    range_x = max(x-1,0), min(x+1+1,N)
    range_y = max(y-1,0), min(y+1+1,N)
    print( x, y, list(range(range_x[0], range_x[1])), list(range(range_y[0],range_y[1])))
    for i in range(range_x[0], range_x[1], 1):
        for j in range(range_y[0], range_y[1], 1):
            if input_image[i,j] == 1 and mark_image[i,j] ==0:
                stack_results.append((i,j))
    return stack_results


def count_warrior_2(input_image):
    N = input_image.shape[0]
    mark_image = np.zeros_like(input_image, dtype=np.int32)

    counter = 1
    for i in range(N):
        for j in range(N):
            if input_image[i, j] == 1 and mark_image[i,j] == 0:
                mark_image[i,j] = counter
                stack_i_j = look_around(i,j, input_image, mark_image)
                while len(stack_i_j) != 0:
                    item = stack_i_j.pop()
                    mark_image[item[0],item[1]] = counter
                    stack_i_j_sub = look_around(item[0],item[1],input_image,mark_image )
                    stack_i_j+=stack_i_j_sub
                counter+=1
    return mark_image, counter-1

if __name__ == "__main__":

    # N = int(raw_input())
    # input_image = np.zeros([N,N], dtype=np.int32)
    # for i in range(N):
    #     row_i = str(raw_input())
    #     for j in range(len(row_i)):
    #         if row_i[j] == "1":
    #             input_image[i,j] = 1

    N = 6
    input_image = np.zeros([N, N], dtype=np.int32)
    strings = ["110110",
               "001001",
               "001101",
               "110001",
               "00000",
               "010100"]
    # strings = ["100110",
    #            "001001",
    #            "000101",
    #            "010001",
    #            "000000",
    #            "010100"]
    # strings = ["100100",
    #            "000110",
    #            "010000",
    #            "010000",
    #            "001000",
    #            "010100"]
    ajact_matrix = {}
    for i in range(N):
        row_i = strings[i]
        for j in range(len(row_i)):
            if row_i[j] == "1":
                input_image[i,j] = 1

    mark_image, counter = count_warrior_2(input_image)
    print(mark_image)
    print (counter)
