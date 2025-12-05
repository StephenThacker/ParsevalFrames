import ParsevalFrames
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations, permutations
from scipy.special import comb 

def periodize_function_1d(unperiodized_function, number_ouput_coefficients,number_of_stack_layers):
    if len(unperiodized_function)%number_ouput_coefficients != 0:
        raise ValueError("unperiodized function length is undivisible by desired output length.")
    
    periodized_function = np.zeros(number_ouput_coefficients)

    for i in range(0,number_of_stack_layers):
        periodized_function += unperiodized_function[i*number_ouput_coefficients:(i+1)*number_ouput_coefficients]
    
    periodized_function /= number_of_stack_layers

    return periodized_function

def gabor_cosine(x,sigma=1,lamb = 2, psi = 0):
    return np.exp((-1*x**2)/(2*sigma**2))*np.cos(2*np.pi*(x/lamb)+psi)
    

def gabor_sine(x,sigma=1,lamb = 2, psi = 0):
    return np.exp((-1*x**2)/(2*sigma**2))*np.sin(2*np.pi*(x/lamb)+psi)

#returns list of functions, list of coefficients for binomial spline expansion
#((a*x+b*y)/denom)^{n}, where denom is the greatest common denominator, a,b are numerical
#coefficients of the variables in the expansion
#We assume that the function has the form
#e^{arg*pi*xi}, where arg1, arg2 are the coefficients of the exponentials
#The point of this is to create a binomial expansion for low pass filter, to compute loss pass filter coefficients and function
#for rapid prototyping of low pass filter.
def binomial_expansion_coefficients(a,b,arg1,arg2,denom,n):
    coefficient_array = []
    arg_1_array = []
    arg_2_array = []
    if arg1 == 0:
        arg_1_array[:] = [0]*(n+1)
    if arg2 == 0:
        arg_2_array[:] = [0]*(n+1)
    #binom expansion goes from 0,...,n, including coefficient n
    denom_factor = denom**n

    for i in range(0,n+1):
        coefficient_array += [int(comb(n,i)*(a**(n-i))*(b**i))]

    if len(arg_1_array) == 0:
        arg_1_array[:] = [arg1*i for i in range(0,n+1)]
    if len(arg_2_array) == 0:
        arg_2_array[:] = [arg2*(n-i) for i in range(0,n+1)]


    total_arg = [a1 + a2 for a1, a2 in zip(arg_1_array, arg_2_array)]
    
    binomial_sum = lambda x: (1/denom_factor)*sum(coefficient_array[k]*np.exp(total_arg[k]*np.pi*1j*x) for k in range(0,n+1))

    results_dict = {"denom": denom_factor,"binom_func":binomial_sum,"int_coef":coefficient_array,"exp_coef":total_arg}

    return results_dict
    


if __name__ == "__main__":

    sigma_1 = 1
    sigma_2 = 2
    sigma_3 = 3

    Total_sigma_1 = sigma_1*3
    Total_sigma_2 = sigma_2*3
    Total_sigma_3 = sigma_3*3

    number_of_points = 50
    number_of_stacks = 12

    x_vector_1 = np.linspace(start= -1*Total_sigma_1, stop=Total_sigma_1,num= number_of_points*number_of_stacks)
    low_pass_test = binomial_expansion_coefficients(1,1,2,0,2,1)
    for key in low_pass_test:
        print(key,low_pass_test[key])

    low_pass = low_pass_test["binom_func"](x_vector_1)
    print("lowpass",low_pass)
    plt.plot(x_vector_1,low_pass)
    plt.show()



    x_vector_2 = np.linspace(start= -1*Total_sigma_2, stop=Total_sigma_2,num= number_of_points*number_of_stacks)
    x_vector_3 = np.linspace(start= -1*Total_sigma_3, stop=Total_sigma_3,num= number_of_points*number_of_stacks)

    gabor_vec_cos_1 = gabor_cosine(x_vector_1)
    gabor_vec_sine_1 = gabor_sine(x_vector_1)

    x_vector_periodized_1 = np.linspace(start=-1*Total_sigma_1, stop = Total_sigma_1, num = number_of_points)
    
    
    plt.plot(x_vector_1,gabor_vec_cos_1)
    plt.show()
    plt.plot(x_vector_1,gabor_vec_sine_1)
    plt.show()
    
    periodized_function_cos_1 = periodize_function_1d(gabor_vec_cos_1,number_of_points,number_of_stacks)
    periodized_function_sin_1 = periodize_function_1d(gabor_vec_sine_1,number_of_points,number_of_stacks)

    plt.plot(x_vector_periodized_1,periodized_function_cos_1)
    plt.show()
    plt.plot(x_vector_periodized_1, periodized_function_sin_1)
    plt.show()
