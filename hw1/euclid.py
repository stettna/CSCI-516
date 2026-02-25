import math

def main():
    x_a  = [1,-3,5]
    x_b = [-2,4,-6]

    sum = 0

    for j in range (0,len(x_a)):
        sum += (x_a[j] - x_b[j])*(x_a[j] - x_b[j])
        
    euclid_dist = math.sqrt(sum)
    print("Euclid. Distance = ", euclid_dist)

main()
    