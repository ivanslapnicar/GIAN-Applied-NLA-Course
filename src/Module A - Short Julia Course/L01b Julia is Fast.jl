### A Pluto.jl notebook ###
# v0.12.21

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

# ╔═╡ 240e99da-c557-43b8-8ecc-f09542591f93
begin
	using Plots
	using PlutoUI
end

# ╔═╡ d54b3cd3-c52c-4db9-9267-2e8871cf4597
md"""
# Julia is Fast


In this notebook, we demonstrate how fast `Julia` is, compared to other dynamically typed languages. 

## Prerequisites

Read [Performance tips](https://docs.julialang.org/en/v1/manual/performance-tips/#man-performance-tips/) section of the `Julia` manual. (20 min) 

## Competences 

The reader should understand effects of "[just-in-time
compiler](https://en.wikipedia.org/wiki/Just-in-time_compilation)"
called [LLVM](http://llvm.org/) on the speed of execution of programs. 
The reader should be able to write simple, but fast, programs containing loops.

## Credits 

Some examples are taken from [The Julia Manual](https://docs.julialang.org/en/v1/).

"""

# ╔═╡ b8c1b263-a5c5-45a2-afdc-744bdc25ca54
md"""
## Summing integer halves

Consider the function `f` which sums halves of integers from `1` to `n`:

__N.B.__ Line number is displayed by bringing pointer over respective dot.
"""

# ╔═╡ 23d162f9-184b-415a-87a2-ccf75b6d5e9a
function f(n)
    s = 0
    for i = 1:n
        s += i/2
    end
    s
end

# ╔═╡ bb8a280d-b118-4842-9220-9382a5408ed5
md"""
In order for the fast execution, the function must first be compiled. Compilation is performed automatically, when the function is invoked for the first time. Therefore, the first call can be done with some trivial choice of parameters.

The timing can be done by macro `@time`: 
"""

# ╔═╡ e27d795b-7d7d-4e76-9ede-9e33b902c2cf
@time f(1)

# ╔═╡ 9042a5c9-97bf-4177-9585-25c4bcbd611f
md"""
Let us now run the big-size computation. Notice the unnaturally high byte allocation and the huge amount of time spent on 
[garbage collection](http://en.wikipedia.org/wiki/Garbage_collection_%28computer_science%29).
"""

# ╔═╡ a9691c4d-7f2c-461c-a42d-cfb5209f3a2d
# Notice the unnaturally high byte  allocation!
@time f(1_000_000)

# ╔═╡ e5545357-9216-46f4-98d1-dfd4a3f31ff7
md"""
Since your computer can execute several _Gigaflops_ (floating-point operations per second), this is rather slow. This slowness is due to _type instability_: variable `s` is in the beginning assumed to be of type `Integer`, while at every other step, the result is a real number of type `Float64`. Permanent checking of types requires permanent memory allocation and deallocation (garbage collection). This is corrected by very simple means: just declare `s` as a real number, and the execution is more than 10 times faster with almost no memory allocation (and, consequently, no garbage collection).
"""

# ╔═╡ 6a9fd473-7939-42f6-a5e4-53dc53abe15f
function f₁(n)
    s = 0.0
    for i = 1:n
        s += i/2
    end
    s
end

# ╔═╡ 93e801a3-c255-428f-878c-60c34d0d7870
@time f₁(1)

# ╔═╡ 401c72ac-07e7-4ed7-9735-e2d8c846633d
@time f₁(1_000_000)

# ╔═╡ 1ec9ad17-67b6-4778-abbd-e31b4d2b2887
md"""
`@time` can alo be invoked as a function:
"""

# ╔═╡ 557aeae6-93cb-419b-88d5-274018a95d84
@time(f₁(1_000_000))

# ╔═╡ 8b754531-102c-4d8f-905d-afd44a354aeb
@time s₂=f₁(1_000_000)

# ╔═╡ a3ca5d12-c330-488b-8639-ad36bc98d3e7
@time(s₃=f₁(1000000))

# ╔═╡ 6571ae43-6583-47f9-bd6a-abe1d3a0f85a
md"""
## Exponential moving average

[Exponential moving average](http://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average) is a fast _one pass_ formula (each data point of the given data set $A$ is accessed only once) often used in high-frequency on-line trading (see [Online Algorithms in High-Frequency Trading](http://cacm.acm.org/magazines/2013/10/168184-online-algorithms-in-high-frequency-trading/fulltext) for more details). __Notice that the output array $X$ is declared in advance.__

Using `return` in the last line is optional.

"""

# ╔═╡ 236594df-0138-4bdd-be1e-bd6a4f69fa02
function fexpma( A::Vector{T}, α::T ) where T
# Fast exponential moving average: X - moving average, 
# A - data, α - exponential forgetting parameter
    n = length(A)
    X = Array{T}(undef,n) # Declare X
    β = one(T)-α
    X[1] = A[1]
    for k = 2:n
        X[k] = β*A[k] + α*X[k-1]
    end
    return X
end


# ╔═╡ 692f908f-8b0b-4809-bafe-d64840650a52
# First run for compilation
fexpma([1.0],0.5)

# ╔═╡ 20fd5340-7aae-4d2b-9da9-27fe15dd1789
md"""
We now generate some big-size data:
"""

# ╔═╡ f12d97f8-498c-4a64-98ec-73486f875728
# Big random slightly increasing sequence
A=[rand() + 0.00001*k*rand() for k=1:20_000_001]

# ╔═╡ 38fad48c-2bc4-433b-80c7-b685f3076691
@time fexpma(A,0.9)

# ╔═╡ 94ce85a1-f6bc-483e-a3b3-70f8e4397e54
md"""
## `@inbounds`

The `@inbounds` command eliminates array bounds checking within expressions. Be certain before doing this. If the subscripts are ever out of bounds, you may suffer crashes or silent corruption. The following program runs a little faster:
"""

# ╔═╡ 02bed44e-1265-4394-87ba-e3e2c617067b
function fexpma₁( A::Vector{T}, α::T ) where T
# Fast exponential moving average: X - moving average, 
# A - data, alpha - exponential forgetting parameter
    n = length(A)
    X = Array{T}(n) # Declare X
    β = one(T)-α
    X[1] = A[1]
    @inbounds for k = 2:n
        X[k] = β*A[k] + α*X[k-1]
    end
    return X
end

# ╔═╡ 328f3f81-4894-4b76-8d49-274fde7da898
@time X=fexpma(A,0.9)

# ╔═╡ 9d7a39c7-acc1-4d40-878d-32a367596d81
md"""
Similar `Matlab` programs run for 3 seconds _without_ prior declaration of $X$, and 0.3 seconds _with_ prior declaration.
"""

# ╔═╡ 2cf93c7e-aa9e-4c0f-8ea5-7e5985e38632
md"""
## Plotting the moving average

Let us plot the data $A$ and its exponential moving average $X$. The dimension of the data is too large for meaningful direct plot, so we use the slider.
"""

# ╔═╡ 3df43da0-7c40-11eb-05dd-e59780fd804b
@bind k Slider(1:1000:length(A)-1,show_value=true)

# ╔═╡ 07a499e7-6399-42e1-9677-a00d5bf95970
scatter(collect(k:k+1000),[A[k:k+1000],X[k:k+1000]],label=["Data" "Moving average"],ms=2, markerstrokecolor=:white)

# ╔═╡ 44a8c386-f5fe-497e-b8b3-829c575d9a11
md"""
## Memory access

The following example is from [Access arrays in memory order, along columns](https://docs.julialang.org/en/stable/manual/performance-tips/#Access-arrays-in-memory-order,-along-columns-1).

Multidimensional arrays in Julia are stored in column-major order, which means that arrays are stacked one column at a time. This convention for ordering arrays is common in many languages like Fortran, Matlab, and R (to name a few). The alternative to column-major ordering is row-major ordering, which is the convention adopted by C and Python (numpy) among other languages. The ordering can be verified using the `vec()` function or the syntax `[:]`:
"""

# ╔═╡ 2222cebe-a13c-4989-8cf1-cf42cbfc6252
B = rand(0:9,4,3)

# ╔═╡ 13ebee2e-0798-4d39-a1d3-20136077f142
B[:]

# ╔═╡ 0aa4c3aa-0cfd-4711-85df-f4b9dbb88c24
vec(B)

# ╔═╡ 638db6e4-cd90-45d4-9418-2c16b45fad73
md"""
The ordering of arrays can have significant performance effects when looping over arrays. Loops should be organized such that the subsequent accessed elements are close to each other in physical memory.
"""

# ╔═╡ 68ff1c7e-7c41-11eb-23a4-c9378daf6e4c
x=rand(5)

# ╔═╡ 72a5f790-7c41-11eb-0271-e91597fa12cd
repeat(x,10)

# ╔═╡ e0f6ef0e-7c41-11eb-374d-6ff8a264b832
repeat(x,1,5)

# ╔═╡ be0c7cde-7c41-11eb-299a-eb97c00e5791
repeat("ha",3)

# ╔═╡ Cell order:
# ╟─d54b3cd3-c52c-4db9-9267-2e8871cf4597
# ╟─b8c1b263-a5c5-45a2-afdc-744bdc25ca54
# ╠═23d162f9-184b-415a-87a2-ccf75b6d5e9a
# ╟─bb8a280d-b118-4842-9220-9382a5408ed5
# ╠═e27d795b-7d7d-4e76-9ede-9e33b902c2cf
# ╟─9042a5c9-97bf-4177-9585-25c4bcbd611f
# ╠═a9691c4d-7f2c-461c-a42d-cfb5209f3a2d
# ╟─e5545357-9216-46f4-98d1-dfd4a3f31ff7
# ╠═6a9fd473-7939-42f6-a5e4-53dc53abe15f
# ╠═93e801a3-c255-428f-878c-60c34d0d7870
# ╠═401c72ac-07e7-4ed7-9735-e2d8c846633d
# ╟─1ec9ad17-67b6-4778-abbd-e31b4d2b2887
# ╠═557aeae6-93cb-419b-88d5-274018a95d84
# ╠═8b754531-102c-4d8f-905d-afd44a354aeb
# ╠═a3ca5d12-c330-488b-8639-ad36bc98d3e7
# ╟─6571ae43-6583-47f9-bd6a-abe1d3a0f85a
# ╠═236594df-0138-4bdd-be1e-bd6a4f69fa02
# ╠═692f908f-8b0b-4809-bafe-d64840650a52
# ╟─20fd5340-7aae-4d2b-9da9-27fe15dd1789
# ╠═f12d97f8-498c-4a64-98ec-73486f875728
# ╠═38fad48c-2bc4-433b-80c7-b685f3076691
# ╟─94ce85a1-f6bc-483e-a3b3-70f8e4397e54
# ╠═02bed44e-1265-4394-87ba-e3e2c617067b
# ╠═328f3f81-4894-4b76-8d49-274fde7da898
# ╟─9d7a39c7-acc1-4d40-878d-32a367596d81
# ╟─2cf93c7e-aa9e-4c0f-8ea5-7e5985e38632
# ╠═240e99da-c557-43b8-8ecc-f09542591f93
# ╠═3df43da0-7c40-11eb-05dd-e59780fd804b
# ╠═07a499e7-6399-42e1-9677-a00d5bf95970
# ╟─44a8c386-f5fe-497e-b8b3-829c575d9a11
# ╠═2222cebe-a13c-4989-8cf1-cf42cbfc6252
# ╠═13ebee2e-0798-4d39-a1d3-20136077f142
# ╠═0aa4c3aa-0cfd-4711-85df-f4b9dbb88c24
# ╟─638db6e4-cd90-45d4-9418-2c16b45fad73
# ╠═68ff1c7e-7c41-11eb-23a4-c9378daf6e4c
# ╠═72a5f790-7c41-11eb-0271-e91597fa12cd
# ╠═e0f6ef0e-7c41-11eb-374d-6ff8a264b832
# ╠═be0c7cde-7c41-11eb-299a-eb97c00e5791
