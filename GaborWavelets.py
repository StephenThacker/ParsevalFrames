import ParsevalFrames
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations, permutations
from scipy.special import comb 
from scipy import signal

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

def one_d_double_convolution_reconstruction(func_vec,filters):
    function_array = []
    reconstructed_function = np.zeros(len(func_vec))
    for i in range(0,len(filters)):
        fconv = signal.convolve(func_vec,filters[i,:],mode="same")
        fconv2 = signal.convolve(fconv,filters[i,:],mode="same")
        reconstructed_function += fconv2


    return {"function_decomp" :function_array, "reconstr_func" : reconstructed_function, "orig_funct" : func_vec, "error": np.linalg.norm(func_vec-reconstructed_function)}

def empirical_reconstruction_error(original_function,reconstr_function):
    return np.linalg.norm(original_function-reconstr_function)



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


    low_pass_filter_dict = binomial_expansion_coefficients(1,1,2,0,2,16)
    print("low pass filter coef")
    print(np.array(low_pass_filter_dict["int_coef"]))
    print("denom")
    print(low_pass_filter_dict["denom"])
    low_pass_filter = np.array(low_pass_filter_dict["int_coef"])*(1/low_pass_filter_dict["denom"])
    x_vector = np.linspace(start = -9,stop = 9, num = 17)

    sigma1 = 1
    sigma2 = 2
    sigma3 = 3

    lambda1 = 0.1
    lambda2 = 0.2
    lambda3 = 0.3
    lambda4 = 0.4

    psi1 = 0
    psi2 = np.pi / 4

    # Define x arrays
    x1 = np.linspace(-3, 3, 17)
    x2 = np.linspace(-6, 6, 17)
    x3 = np.linspace(-9, 9, 17)

    # Generate cosine filters (g1 to g8)
    g1 = np.exp(-x1**2 / (2 * sigma1**2)) * np.cos(2 * np.pi * x1 / lambda1 + psi1)
    g2 = np.exp(-x1**2 / (2 * sigma1**2)) * np.cos(2 * np.pi * x1 / (lambda1 - 0.2) + psi2)
    g3 = np.exp(-x1**2 / (2 * (sigma1 - 0.5)**2)) * np.cos(2 * np.pi * x1 / lambda2 + psi1)
    g4 = np.exp(-x1**2 / (2 * sigma1**2)) * np.cos(2 * np.pi * x1 / lambda2 + psi2)
    g5 = np.exp(-x1**2 / (2 * sigma1**2)) * np.cos(2 * np.pi * x1 / lambda3 + psi1)
    g6 = np.exp(-x1**2 / (2 * sigma1**2)) * np.cos(2 * np.pi * x1 / lambda3 + psi2)
    g7 = np.exp(-x1**2 / (2 * sigma1**2)) * np.cos(2 * np.pi * x1 / lambda4 + psi1)
    g8 = np.exp(-x1**2 / (2 * sigma1**2)) * np.cos(2 * np.pi * x1 / lambda4 + psi2)

    # Generate sine filters (h1 to h32)
    h1 = np.exp(-x2**2 / (2 * sigma2**2)) * np.sin(2 * np.pi * x2 / lambda1 + psi1)
    h2 = np.exp(-x2**2 / (2 * sigma2**2)) * np.sin(2 * np.pi * x2 / lambda1 + psi2)
    h3 = np.exp(-x2**2 / (2 * sigma2**2)) * np.sin(2 * np.pi * x2 / lambda2 + psi1)
    h4 = np.exp(-x2**2 / (2 * sigma2**2)) * np.sin(2 * np.pi * x2 / lambda2 + psi2)
    h5 = np.exp(-x2**2 / (2 * sigma2**2)) * np.sin(2 * np.pi * x2 / lambda3 + psi1)
    h6 = np.exp(-x2**2 / (2 * sigma2**2)) * np.sin(2 * np.pi * x2 / lambda3 + psi2)
    h7 = np.exp(-x2**2 / (2 * sigma2**2)) * np.sin(2 * np.pi * x2 / lambda4 + psi1)
    h8 = np.exp(-x2**2 / (2 * sigma2**2)) * np.sin(2 * np.pi * x2 / lambda4 + psi2)
    h9 = np.exp(-x3**2 / (2 * sigma3**2)) * np.sin(2 * np.pi * x3 / lambda1 + psi1)
    h10 = np.exp(-x3**2 / (2 * sigma3**2)) * np.sin(2 * np.pi * x3 / lambda1 + psi2)
    h11 = np.exp(-x3**2 / (2 * sigma3**2)) * np.sin(2 * np.pi * x3 / lambda2 + psi1)
    h12 = np.exp(-x3**2 / (2 * sigma3**2)) * np.sin(2 * np.pi * x3 / lambda2 + psi2)
    h13 = np.exp(-x3**2 / (2 * sigma3**2)) * np.sin(2 * np.pi * x3 / lambda3 + psi1)
    h14 = np.exp(-x3**2 / (2 * sigma3**2)) * np.sin(2 * np.pi * x3 / lambda3 + psi2)
    h15 = np.exp(-x3**2 / (2 * sigma3**2)) * np.sin(2 * np.pi * x3 / lambda4 + psi1)
    h16 = np.exp(-x3**2 / (2 * sigma3**2)) * np.sin(2 * np.pi * x3 / lambda4 + psi2)
    h17 = np.exp(-x3**2 / (2 * sigma3**2)) * np.sin(2 * np.pi * x3 / lambda1 + psi1)
    h18 = np.exp(-x3**2 / (2 * sigma3**2)) * np.sin(2 * np.pi * x3 / lambda1 + psi2)
    h19 = np.exp(-x3**2 / (2 * sigma3**2)) * np.sin(2 * np.pi * x3 / lambda2 + psi1)
    h20 = np.exp(-x3**2 / (2 * sigma3**2)) * np.sin(2 * np.pi * x3 / lambda2 + psi2)
    h21 = np.exp(-x3**2 / (2 * sigma3**2)) * np.sin(2 * np.pi * x3 / lambda3 + psi1)
    h22 = np.exp(-x3**2 / (2 * sigma3**2)) * np.sin(2 * np.pi * x3 / lambda3 + psi2)
    h23 = np.exp(-x3**2 / (2 * sigma3**2)) * np.sin(2 * np.pi * x3 / lambda4 + psi1)
    h24 = np.exp(-x3**2 / (2 * sigma3**2)) * np.sin(2 * np.pi * x3 / lambda4 + psi2)
    h25 = np.exp(-x3**2 / (2 * sigma3**2)) * np.sin(2 * np.pi * x3 / lambda1 + psi1)
    h26 = np.exp(-x3**2 / (2 * sigma3**2)) * np.sin(2 * np.pi * x3 / lambda1 + psi2)
    h27 = np.exp(-x3**2 / (2 * sigma3**2)) * np.sin(2 * np.pi * x3 / lambda2 + psi1)
    h28 = np.exp(-x3**2 / (2 * sigma3**2)) * np.sin(2 * np.pi * x3 / lambda2 + psi2)
    h29 = np.exp(-x3**2 / (2 * sigma3**2)) * np.sin(2 * np.pi * x3 / lambda3 + psi1)
    h30 = np.exp(-x3**2 / (2 * sigma3**2)) * np.sin(2 * np.pi * x3 / lambda3 + psi2)
    h31 = np.exp(-x3**2 / (2 * sigma3**2)) * np.sin(2 * np.pi * x3 / lambda4 + psi1)
    h32 = np.exp(-x3**2 / (2 * sigma3**2)) * np.sin(2 * np.pi * x3 / lambda4 + psi2)

    # Construct GaborMatrix
    high_pass_filters = np.array([
        g1, g2, g3, g4, g5, g6, g7, g8,
        h1, h2, h3, h4, h5, h6, h7, h8,
        h9, h10, h11, h12, h13, h14, h15, h16,
        h17, h18, h19, h20, h21, h22, h23, h24,
        h25, h26, h27, h28, h29, h30, h31, h32
    ])

    for i in range(0,len(high_pass_filters)):
        high_pass_filters[i] = high_pass_filters[i] - np.mean(high_pass_filters[i])
        

    Bmat, res = ParsevalFrames.funmin(low_pass_filter,high_pass_filters,orthoganalize_projection=False)
    print(type(Bmat))

    print("length")
    print(len(Bmat))

    x= np.linspace(-2,2,24000)
    f = ((x**2)*np.sin(x**-2))/(np.exp((0.01)*(x**2)))
    reconstr_dict = one_d_double_convolution_reconstruction(f,Bmat)


    reconstr_dict_orig = one_d_double_convolution_reconstruction(f,np.vstack((low_pass_filter,high_pass_filters)))

    print("reconstruction error with filters", reconstr_dict["error"]/np.linalg.norm(f))
    print("reconstruction error of original", reconstr_dict_orig["error"])
    #plt.plot(x,reconstr_dict_orig['reconstr_func'],alpha = 0.4)

    plt.plot(x,reconstr_dict["reconstr_func"], alpha = 0.3)

    plt.plot(x,f,alpha = 0.2)
    plt.show()



    """print("\nBmat (scaled by 100):")
    print(Bmat)"""

    print("\nOptimized λ:")
    print(res)




    """
    sigma_1 = 1
    sigma_2 = 2
    sigma_3 = 3

    vector_base = 17
    x_vector = np.linspace(start = -9,stop = 9, num = 17*3)

    const = 20/51
    const /= 51
    const *= 80
    x = -10/51

    scale_vec = np.array([x + const*i for i in range (0,51)])
    scale_vec_1 = np.array(scale_vec[:11,None])
    scale_vec_2 = np.array(scale_vec[11:30,None])
    scale_vec_3 = np.array(scale_vec[30:51,None])

    sigma_vec_1 = np.array([sigma_1 for i in range(0,len(x_vector))])
    sigma_vec_1 = sigma_vec_1[None, :]


    sigma_vec_2 = np.array([sigma_2 for i in range(0,len(x_vector))])
    sigma_vec_2 = sigma_vec_2[None, :]

    sigma_vec_3 = np.array([sigma_3 for i in range(0,len(x_vector))])
    sigma_vec_3 = sigma_vec_3[None, :]

    #modifies x-vector to allow for shape broadcasting
    #Takes (1,N) matrix and broadcasts it to (N_1, 1), resulting in 
    #Matrix of size (N_1,N), where N_1 is related to the shifts and
    #N is the size of the input vector (i.e., low pass filter coeffficients)
    x_vector_dynamic = x_vector[None,:]

    first_array = gabor_cosine(x_vector_dynamic,sigma_vec_1,psi=scale_vec_1)
    second_array = gabor_sine(x_vector_dynamic,sigma_vec_2,psi=scale_vec_2)
    third_array = gabor_sine(x_vector_dynamic,sigma_vec_3, psi = scale_vec_3)

    y_vector_1 = first_array[0,:]

    y_vector_2 = first_array[4,:]
    y_vector_3 = first_array[9,:]

    high_pass_filters = np.vstack((first_array,second_array,third_array))


    #show all filters
    
    for i in range(0,len(first_array)):
        plt.plot(x_vector,first_array[i,:])

    plt.show()

    for i in range(0,len(second_array)):
        plt.plot(x_vector,second_array[i,:])

    plt.show()

    for i in range(0,len(third_array)):
        plt.plot(x_vector,third_array[i,:])

    plt.show()

    print("high pass filters")
    for i in range(0,len(high_pass_filters)):
        print("highpass",i)
        print(high_pass_filters[i])
    

    low_pass_filter_dict = binomial_expansion_coefficients(1,1,2,0,2,50)
    low_pass_filter = np.array(low_pass_filter_dict["int_coef"])*(1/low_pass_filter_dict["denom"])
    print("low pass filter")
    print(len(low_pass_filter))
    #orthogonalized_filter_matrix = orthogonalize_vector_group(high_pass_filters,low_pass_filter)
    Bmat, res = ParsevalFrames.funmin(low_pass_filter,high_pass_filters,orthoganalize_projection=True)
    print("Bmatrix")
    print(Bmat)
    print("res")
    print(res)"""






    '''low_pass = np.array([0,1,1],dtype=float)
    matrix = np.array([[1,1,3],[0,1,2],[1,0,0]],dtype=float)
    orthogonalize_vector_group(matrix,low_pass)

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
    plt.show()'''


