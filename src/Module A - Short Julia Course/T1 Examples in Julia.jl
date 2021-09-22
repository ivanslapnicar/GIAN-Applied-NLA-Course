### A Pluto.jl notebook ###
# v0.16.1

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 3f890c7b-44ad-426a-bf26-32ff17fa8ec4
md"""
# Examples in Julia - Tutorial 1
"""

# ╔═╡ 5bc2c872-fded-4bc8-8044-9eda1822c505
md"""
## Assignment 1

Using the package `Polynomials.jl`, write the function which implements 
[Graeffe's method](https://en.wikipedia.org/wiki/Graeffe%27s_method)
(see also [here](http://mathworld.wolfram.com/GraeffesMethod.html))
for computing roots of polynomials with only real roots with simple moduli.

In the function, use Julia's `BigFloat` numbers to overcome the main disadvantage of the method. What is the number of significant decimal digits, and the largest and the smallest number?

Test the function on the [Wilkinson's polynomial](https://en.wikipedia.org/wiki/Wilkinson%27s_polynomial) $\omega(x)$, and the [Chebyshev polynomial](https://en.wikipedia.org/wiki/Chebyshev_polynomials) $T_{50}(x)$ 
(the latter needs to be transformed in order to apply the method).

Compare your solutions with the exact solutions.
"""

# ╔═╡ 5b542ad3-8471-4adc-9c89-d647000b8ca0
md"""
## Assignment 2

Write the function which computes simple LU factorization (without pivoting) where the matrix is overwritten by the factors.

Make sure that the function also works with block-matrices. 

Compare the speed on standard matrices and block-matrices with the built-in LU factorization (which also uses block algorithm AND pivoting). Check the accuracy.
"""

# ╔═╡ dbd0a150-2ad2-49e3-b585-d34bfc2ff3b8
md"""
## Assignment 3

Use the function `eigvals()` to compute the eigenvalues of $k$ random matrices (with uniform and normal distribution of elements) of order $n$.

Plot the results using the macro `@bind` from the package `PlutoUI.jl`.
Use `Plots.jl` for plotting.

Are the eigenvalues random? Can you describe their behaviour? Can random matrices be used to test numerical algorithms?
"""

# ╔═╡ 4cd45baf-67f6-43cc-a9bf-3a89006c3814


# ╔═╡ Cell order:
# ╟─3f890c7b-44ad-426a-bf26-32ff17fa8ec4
# ╟─5bc2c872-fded-4bc8-8044-9eda1822c505
# ╟─5b542ad3-8471-4adc-9c89-d647000b8ca0
# ╟─dbd0a150-2ad2-49e3-b585-d34bfc2ff3b8
# ╠═4cd45baf-67f6-43cc-a9bf-3a89006c3814
