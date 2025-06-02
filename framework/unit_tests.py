import numpy as np
import states_and_witnesses as sw
import operations as op

def test_get_1_W(rho, counts):
    print(W3_obj(rho=rho, counts=counts).W3_1)

def test_get_W3s(rho, counts):
    print(W3_obj(rho=rho, counts=counts).get_witnesses('operators'))

def test_get_W5s(rho, counts):
    print(W5_obj(rho=rho, counts=counts).get_witnesses('operators'))

if __name__ == '__main__':
    # example state: PHI_P = 1/sqrt(2)*(|HV> + <VH|)
    rho = np.array[[0, 0, 0, 0],
           [0, 0.5, 0.5, 0],
           [0, 0.5, 0.5, 0],
           [0, 0, 0, 0]]
    counts = counts
    W3_obj = sw.W3
    W5_obj = sw.W5

    test_get_1_W()
    test_get_W3s()
    test_get_W5s()