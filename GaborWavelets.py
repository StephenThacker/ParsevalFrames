import ParsevalFrames
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations, permutations
from scipy.special import comb 
from scipy import signal
from scipy.optimize import minimize_scalar

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

def single_filter_reconstruction_error(func_vec, filter):
    f_conv = signal.convolve(func_vec,filter, mode = "same")
    reconstructed_func = signal.convolve(f_conv,filter,mode = 'same')

    return empirical_reconstruction_error(func_vec,reconstructed_func)

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

#sigma_vector is a single parameter
#lambda_vector and psi_vector are lists which determine the parameters of the functions and
#are assumed to be the same length.
def define_cosine_filters(x_vector,sigma_vector,lambda_vector,psi_vector):
    coefficient_list = []

    for i in range(0,len(lambda_vector)):
        coefficient_list += [np.exp(-x_vector**2 / (2 * sigma_vector**2)) * np.cos(2 * np.pi * x_vector / lambda_vector[i] + psi_vector[i])]

    return coefficient_list

#Defines functions for optimization. Defining this as a separate function for clarity
def define_cosine_functions(x_vector,sigma_vector,psi_vector):
    function_list = []

    for i in range(0,len(psi_vector)):
        function_list += [lambda lambda_coeff, psi_vec = psi_vector[i] : np.exp(-x_vector**2 / (2 * sigma_vector**2)) * np.cos(2 * np.pi * x_vector / lambda_coeff + psi_vec)]

    return function_list


def define_sine_filters(x_vector,sigma_vector,lambda_vector, psi_vector):
    coefficient_list = []
    for i in range(0,len(lambda_vector)):
        coefficient_list += [np.exp(-x_vector**2 / (2 * sigma_vector**2)) * np.sin(2 * np.pi * x_vector / lambda_vector[i] + psi_vector[i])]

    return coefficient_list

def define_sine_functions(x_vector,sigma_vector, psi_vector):
    function_list = []
    for i in range(0,len(psi_vector)):
        function_list += [lambda lambda_coeff, psi_vec = psi_vector[i]: np.exp(-x_vector**2 / (2 * sigma_vector**2)) * np.sin(2 * np.pi * x_vector / lambda_coeff + psi_vec)]

    return function_list


#Greedy algorithm to preselect which filters have the best chance of reconstructing the image.
#Step1: identify which filter has the largest contribution (i.e, the smallest reconstruction error, maybe largest norm contr.)
#Identify which parameter optimizes that filter
#Remove filter and parameter from list
#Continue until no more filters
#Need to use a fast algorithm because total possibilities are Num_of_filters^{Num of Paramaters}, which is likely larger
#than the number of all atoms in the universe.
def preselect_filters(filter_matrix, targ_function, lambda_universe, filter_func_list):

    def reconstruction_error(high_pass_filt):
        f_conv = signal.convolve(targ_function,high_pass_filt, mode = "same")
        reconstructed_func = signal.convolve(f_conv,high_pass_filt,mode = 'same')
        error = np.linalg.norm(targ_function - reconstructed_func)

        return error
    
    def reconstructed_norm(high_pass_filt):
        f_conv = signal.convolve(targ_function,high_pass_filt, mode = "same")
        reconstructed_func = signal.convolve(f_conv,high_pass_filt,mode = 'same')
        norm = np.linalg.norm(reconstructed_func)

        return norm
    
    recon_norms = [(filt, reconstructed_norm(filt),func) for filt,func in zip(filter_matrix,filter_func_list)]
    norm_sorted_filts = sorted(recon_norms, key = lambda tup:tup[1], reverse=True)

    sorted_filters_only = [tup[0] for tup in norm_sorted_filts]

    remaining_filters = norm_sorted_filts
    #greedy algorithm
    optimized_filters = []
    resolved_coeffs = []
    for i in range(0,len(remaining_filters)):
        coeff, norm, func = remaining_filters[i]
        coeff_list = []
        for j in range(0,len(lambda_universe)):
            coeff_list += [(reconstructed_norm(func(lambda_universe[j])),func(lambda_universe[j]),j,lambda_universe[j])]
        coeff_list = sorted(coeff_list, key = lambda tup: tup[0])
        j_index = coeff_list[-1][2]
        lambda_universe.pop(j_index)
        optimized_filters += [coeff_list[-1][1]]
        resolved_coeffs += [coeff_list[-1][-1]]

    print("optimization coefficients")
    print(resolved_coeffs)

    print("optimized filters")
    print(np.array(optimized_filters).shape)


    return np.array(optimized_filters)


#Idea:
#Step 1: Find the filter that maximizes the norm of the convolution w/ signal
#Step 2, find the dilation/ shift parameters that minimize the reconstruction error with respect to a single filter
#Step 3: Find the coefficient that minimizes the residual norm of the convolved function and the dilated/shifted function
#Step 4: Subtract from the residual, Iterate this process until all filters are been optimized.
#Idea: Should we also consider a way to do this where this process is somehow automated.
#Is there a way to do it where We just give it Sin and Cos and it finds the right combination that optimizes the selections?
def preselect_filters_OBP(filter_matrix, targ_function, lambda_universe, filter_func_list):

    def convolution_norm(high_pass_filt):
        f_conv = signal.convolve(targ_function,high_pass_filt, mode = "same")

        return np.linalg.norm(f_conv)
    
    def reconstruction_norm_scaled(high_pass_filt,func):
        f_conv = signal.convolve(func,high_pass_filt, mode = "same")
        f_double = signal.convolve(f_conv,high_pass_filt, mode = "same")
        normalized_f = f_double/np.linalg.norm(high_pass_filt)**2
        resid_func = func - normalized_f
        resid_num = np.linalg.norm(func - normalized_f)
        return [resid_func, resid_num]
    
    def coefficient_optimization_scalar(alpha,current_resid,high_pass_filt):
        scaled_high_pass = alpha*high_pass_filt
        f_conv = signal.convolve(current_resid,scaled_high_pass, mode = "same")
        f_double = signal.convolve(f_conv,scaled_high_pass, mode = "same")
        new_residual = current_resid - f_double
        return np.linalg.norm(new_residual)
    
    def calculate_resid(current_resid,high_pass_filt):
        f_conv = signal.convolve(current_resid,high_pass_filt, mode = "same")
        f_double = signal.convolve(f_conv,high_pass_filt, mode = "same")
        new_residual = current_resid - f_double
        return new_residual
    
    recon_norms = [(filt, convolution_norm(filt),func) for filt,func in zip(filter_matrix,filter_func_list)]
    norm_sorted_filts = sorted(recon_norms, key = lambda tup:tup[1], reverse=True)

    #greedy algorithm
    new_high_pass_list = []
    resid = targ_function
    for i in range(0,len(norm_sorted_filts)):
        max_filt_lambda = []
        for j in range(0,len(lambda_universe)):
            lamb_j = lambda_universe[j]            
            #retrives the highpass filter associated with lambda[j]
            coeff_j = norm_sorted_filts[i][2](lamb_j)
            resid_func, resid_num = reconstruction_norm_scaled(coeff_j,resid)
            max_filt_lambda.append((coeff_j, resid_func, resid_num,j))
        max_filt_lambda = sorted(max_filt_lambda, key = lambda tup: tup[2])
        unscaled_new_high_pass, resid_func,resid_num,lambda_j = max_filt_lambda[0]
        lambda_universe.pop(lambda_j)
        print("max_lambda")
        print(resid_num)

        #Need to add sub-routine here that optimizes the scalar for the projection

        optimiz_result = minimize_scalar(coefficient_optimization_scalar,args=(resid,unscaled_new_high_pass))
        alpha = optimiz_result.x
        scaled_new_high_pass = alpha*unscaled_new_high_pass
        print("alpha")
        print(alpha)
        new_high_pass_list.append(unscaled_new_high_pass)
        adjusted_resid = calculate_resid(resid,scaled_new_high_pass)
        resid = adjusted_resid

    return np.array(new_high_pass_list)

    """f_conv = signal.convolve(targ_function,sorted_filters_only[0], mode = "same")
    reconstructed_func = signal.convolve(f_conv,sorted_filters_only[0],mode = 'same')

    plt.plot(x_vector ,reconstructed_func, alpha = 0.3)

    plt.plot(x_vector,function,alpha = 0.2)
    plt.show()

    return"""

'''
class GaborFilter():

    def __init(self, x_arrays, sigma_array, lambda_array,psi_array):'''






if __name__ == "__main__":


    low_pass_filter_dict = binomial_expansion_coefficients(1,1,2,0,2,16)
    low_pass_filter = np.array(low_pass_filter_dict["int_coef"])*(1/low_pass_filter_dict["denom"])
    x_vector = np.linspace(start = -9,stop = 9, num = 17)

    sigma1 = 1
    sigma2 = 2
    sigma3 = 3

    lambda_vec_init = [0.01,0.2,0.3,0.4]
    psi_vec_init = [0, np.pi/4]

    cos_lambda_array = [x for x in lambda_vec_init for _ in range(2)]
    cos_psi_array = psi_vec_init*4
        
    sin_lambda_array_1 = [x for x in lambda_vec_init for _ in range(2)]
    sin_psi_array_1 = psi_vec_init*4

    sin_lambda_array_2 = [x for x in lambda_vec_init for _ in range(6)]
    sin_psi_array_2 = psi_vec_init*12

    # Define x arrays
    x1 = np.linspace(-3, 3, 17)
    x2 = np.linspace(-6, 6, 17)
    x3 = np.linspace(-9, 9, 17)

    lambda_step  = 0.05

    lambda_universe = [lambda_step*i + 0.5*lambda_vec_init[0] for i in range(400)]


    cos_filter_bank = define_cosine_filters(x1,sigma1,cos_lambda_array,cos_psi_array)
    cos_filter_functions = define_cosine_functions(x1,sigma1,cos_psi_array)
    sin_filter_bank_1 = define_sine_filters(x2,sigma2, sin_lambda_array_1, sin_psi_array_1)
    sin_filter_functions1 = define_sine_functions(x2,sigma2,sin_psi_array_1)
    sin_filter_bank_2 = define_sine_filters(x3,sigma3, sin_lambda_array_2,sin_psi_array_2)
    sin_filter_functions2 = define_sine_functions(x3, sigma3, sin_psi_array_2)

    high_pass_filters = np.vstack([cos_filter_bank,sin_filter_bank_1,sin_filter_bank_2])
    print("high pass filters len")
    print(len(high_pass_filters))
    high_pass_functions = cos_filter_functions+sin_filter_functions1+sin_filter_functions2

    x= np.linspace(-2,2,24000)
    f = ((x**2)*np.sin(x**-2))/(np.exp((0.01)*(x**2)))

    for i in range(0,len(high_pass_filters)):
        high_pass_filters[i] = high_pass_filters[i] - np.mean(high_pass_filters[i])

    
    high_pass_filters = preselect_filters_OBP(high_pass_filters, f, lambda_universe, high_pass_functions)

    for i in range(0,len(high_pass_filters)):
        high_pass_filters[i] = high_pass_filters[i] - np.mean(high_pass_filters[i])
        

    Bmat, res = ParsevalFrames.funmin(low_pass_filter,high_pass_filters,orthoganalize_projection=False)


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


    plt.plot(x_vector_periodized_1, periodized_function_sin_1)
    plt.show()'''

