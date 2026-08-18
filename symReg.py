"""
Author: Xavier Grundler and Nicholas Cox
Title: symReg.py
Description: Using the PySR package, find a symbolic regression
             for HIC observables/parameters.
"""

from pysr import PySRRegressor

#this model will create a rational function, with some constraints
def rational(size=20,niter=200,weight=0.000502,choice="accuracy",prog=False):
    #create the model
    model = PySRRegressor(
        maxsize = size, #complexity must stay below this
        niterations = niter, #how long to train for

        #the allowed operators
        binary_operators = ["+", "-", "*", "/"],
        unary_operators = ["square", "cube", "quad(x)=x^4"],

        #define quad(x)
        extra_sympy_mappings = {"quad": lambda x: x**4},

        nested_constraints = {
            "square" : {"square":1, "cube":1, "quad":1},
            "cube" : {"square":1, "cube":0, "quad":0},
            "quad" : {"square":0, "cube":0, "quad":0}
            },

        #determines rate of randomization
        weight_randomize = weight,

        #choose most accurate model
        model_selection = choice,

        #display progress in terminal
        progress = prog
        )

    return model

def modifiedrational(size=20,niter=200,weight=0.000502,choice="accuracy",prog=False):
    #create the model
    model = PySRRegressor(
        maxsize = size, #complexity must stay below this
        niterations = niter, #how long to train for

        #the allowed operators
        binary_operators = ["+", "-", "*", "/"],
        unary_operators = ["square", "cube", "quad(x)=x^4"],

        #define quad(x)
        extra_sympy_mappings = {"quad": lambda x: x**4},

        nested_constraints = {
            "square" : {"square":1, "cube":1, "quad":1},
            "cube" : {"square":1, "cube":0, "quad":0},
            "quad" : {"square":0, "cube":0, "quad":0}
            },

        complexity_of_operators = {"quad":3,"cube":2,"/":2
            },

        #determines rate of randomization
        weight_randomize = weight,

        #choose most accurate model
        model_selection = choice,

        #display progress in terminal
        progress = prog
        )

    return model

def polynomial(size=20,niter=200,weight=0.000502,choice="accuracy",prog=False):
    #create the model
    model = PySRRegressor(
        maxsize = size, #complexity must stay below this
        niterations = niter, #how long to train for

        #the allowed operators
        binary_operators = ["+", "-", "*"],
        unary_operators = ["square", "cube", "quad(x)=x^4"],

        #define quad(x)
        extra_sympy_mappings = {"quad": lambda x: x**4},

        nested_constraints = {
            "square" : {"square":1, "cube":1, "quad":1},
            "cube" : {"square":1, "cube":0, "quad":0},
            "quad" : {"square":0, "cube":0, "quad":0}
            },

        complexity_of_operators = {"quad":3,"cube":2},

        #determines rate of randomization
        weight_randomize = weight,

        #choose most accurate model
        model_selection = choice,

        #display progress in terminal
        progress = prog
        )

    return model

def modifiedpolynomial(size=20,niter=200,weight=0.000502,choice="accuracy",prog=False):
    model = PySRRegressor(
        maxsize = size, #complexity must stay below this
        niterations = niter, #how long to train for

        #the allowed operators
        binary_operators = ["+", "-", "*"],
        unary_operators = ["square", "cube", "identity"],

        extra_sympy_mappings = {
        "square": lambda x: x**2,
        "cube": lambda x: x**3,
        "identity": lambda x: x},

        nested_constraints = {
            "identity" : {"identity":0, "square":0, "cube":0},
            "square" : {"identity":0, "square":0, "cube":0},
            "cube" : {"identity":0, "square":0, "cube":0},
            },

        complexity_of_constants=1,
        
        #determines rate of randomization
        weight_randomize = weight,

        #choose most accurate model
        model_selection = choice,

        #display progress in terminal
        progress = prog
        )

    return model

def allfunctions(size=20,niter=200,weight=0.000502,choice="accuracy",prog=False):
    #create the model
    model = PySRRegressor(
        maxsize = size, #complexity must stay below this
        niterations = niter, #how long to train for
        
        extra_sympy_mappings = {"quad": lambda x: x**4},
        
        #the allowed operators
        binary_operators = ["+","-","*","/"],
        unary_operators = ["square","cube","quad(x)=x^4","sin","cos","exp","log"],
        
        #determines rate of randomization
        weight_randomize = weight,
        
        #choose most accurate model
        model_selection = choice,
        
        progress = prog
        )
    return model


