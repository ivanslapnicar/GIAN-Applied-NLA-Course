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

# ╔═╡ 1685eaa0-7c38-11eb-0dc2-8bfbea75d947
begin
	using PlutoUI
	PlutoUI.TableOfContents(aside=true)
end

# ╔═╡ c75546c0-47ae-4706-9a73-aad97700584d
# Traces of 5 random matrices. We need the package LinearAlgebra
begin
	using LinearAlgebra
	[tr(rand(n,n)) for n=1:5]
end

# ╔═╡ 1f472327-7696-4070-877b-36fa00b32d11
md"""
# Lightning Round - Basic Features and Commands 



In this notebook, we go through basic constructs and commands.

__Competences__ 

The user should know to start `Julia`, how to exit, learn some features and be able to write simple programs.

__Credits__

This notebook is based on the [Julia Tutorial](https://github.com/JuliaLang/julia-tutorial).
"""

# ╔═╡ 5d1910ad-80e4-4ea0-972f-3cb47ae7f057
md"""
# Julia resources

Julia resources are accessible through the [Julia home page](http://julialang.org).

Please check [Packages](https://julialang.org/packages/), [Documentation](https://docs.julialang.org/en/v1/) and JuliaCon where you will also find links to videos from previous conferences.

Julia is open source and all routines are available on GitHub. You will learn how to make full use of this later in the course.

"""

# ╔═╡ 3275dbbd-8612-4395-adb9-1318cc85d136
md"""
# Execution
To execute cell use `Shift + Enter` or press __Run__ button (circled triangle at the bottom right of the cell).

Pluto notebook is __reactive__, so after any change all affected cells are recomputed. 
"""

# ╔═╡ 8f8e9dba-72ba-41e1-b39a-aeef95402555
md"""
## Markdown cells

Possibility to write comments / code / formulas in `Markdown` cells, makes Pluto notebooks ideal for teaching and research. Text is written using _Julia Markdown_, which is _GitHub Markdown_ with additional understanding of basic `LaTeX`.

[Mastering (GitHub) Markdown](https://guides.github.com/features/mastering-markdown/) is a 3-minute read, another short and very good manual is at [link](http://daringfireball.net/projects/markdown/).
"""

# ╔═╡ d9196675-4ec2-4236-a7f1-bcc589e234ab
md"""
## Which version of `Julia` is running?

The output of this command is in the console.
"""

# ╔═╡ 4f2aee7c-6e08-404c-a561-18f1662e5837
versioninfo()

# ╔═╡ b46be41b-edc5-4321-850c-ecb82e13ac7a
md"""
## Quitting

Exiting from `julia>` kernel: `exit()`.

Killing Pluto server `2 × Ctrl C`.
"""

# ╔═╡ dbc23df0-c4f0-4efc-a452-3392e0a18bf1
# exit()

# ╔═╡ 9779cbcd-d500-4383-a7e3-ff909837aaa6
md"""
# Punctuation review

* `[...]` are for indexing, array constructors and __Comprehensions__
* `(...)` are __required__ for functions `quit()`, `tic()`, `toc()`, `help()` 
* `{...}` are for arrays
* `#` is for comments
"""

# ╔═╡ 2b973dd0-1e52-44c9-bf22-be092080fa3e
md"""
# Basic indexing
"""

# ╔═╡ 334ea76b-a126-41b0-8945-ba166507b0a1
# Matrix with random entries between 0 and 1
A₀=rand(5,5)

# ╔═╡ 9ee1a335-8060-4c6c-b1c6-4e1699053086
A₀[1,1]

# ╔═╡ 9293e691-1b04-4989-a891-fbd1bd3306fe
# You can index into output directly
rand(5,5)[1:2,3:4]

# ╔═╡ 2d16ab49-9244-4ef6-b5e0-fa5304b367e4
md"""
## Indexing is elegant

If you want to compute the lower right $2\times 2$ block of $A^{10}$, in most languages you need to first compute $B=A^{10}$ and then index into $B$. In Julia, the command is simply
"""

# ╔═╡ 9e98e5ce-3ca2-4bf8-84f1-23dc3e793b9e
(A₀^10)[4:5,4:5] # Parenthesses around A^10 are necessary

# ╔═╡ 6acd3389-37b1-48a7-9a04-795372503bac
md"""
## Comprehensions - elegant array constructors
"""

# ╔═╡ 2078e00d-b3b2-4188-b5be-9835de2ab357
[i for i=1.0:5]

# ╔═╡ 0bc90ce5-9976-498d-930c-e34fea9c7193
x₀=1:10

# ╔═╡ 3d949c09-801b-434d-a972-f590e8de8704
[ x₀[i]+x₀[i+1] for i=1:9 ]

# ╔═╡ 963ea673-6469-4718-b348-a281578c7742
z = [Matrix{Int64}(I,n,n) for n=1:5]  # z is Array of Arrays

# ╔═╡ ae3fb606-c31d-4edd-9d4c-4c3982c86be4
# First element is a 1x1 Array
z[1]

# ╔═╡ e4790db9-99ee-4200-bc30-d80d0494f23f
# What is the fourth element?
z[4]

# ╔═╡ 9efef8f4-130b-4286-b109-a43e47a1d00b
# Another example of a comprehension
A=[ i+j for i=1:5, j=1:5 ]

# ╔═╡ b80eb1ee-0d74-47fe-8fbf-87140fad8e7c
# Notice the promotion
B=[ i+j for i=1:5, j=1.0:5 ]

# ╔═╡ de3245f1-8b93-4e9e-be8c-76f7f096907f
md"""
# Commands `ndims()` and `typeof()`
"""

# ╔═╡ 6cc8eb48-4dbb-4599-b126-b4b89f8376de
ndims(B)

# ╔═╡ cdff612c-ad32-4fc8-aa11-d4501f490f05
# z is a one-dimensional array
ndims(z)

# ╔═╡ 4725ae25-0800-442e-88b5-19243da2a3e9
# Array of Arrays
typeof(z)

# ╔═╡ 48da687f-935d-4262-ad0a-f5e8f334da9a
# z[5] is a two-dimensional array
typeof(z[5])

# ╔═╡ 0a918be5-070d-4f41-a09d-dc5d3238189b
typeof(A)

# ╔═╡ 84e88328-de83-49ed-89f8-f9ba13f1ea52
md"""
# Vectors are 1-dimensional arrays

See [Multi-dimensional arrays](https://docs.julialang.org/en/v1/manual/arrays/) for more.
"""

# ╔═╡ a232e8d5-1a28-4a85-9cd4-e58bb79b77a0
# This is 2-dimensional array
v=rand(5,1)

# ╔═╡ 851065e9-d701-46ae-9f22-d6ef0975694f
# This is an 1-dimensional array or vector
vv=vec(v) 

# ╔═╡ 029ac818-a082-4454-98fc-b4a9b94b719a
# Notice that they are different
v==vv

# ╔═╡ af54e7ef-3512-4cdf-bea5-e7848299638d
# Again a promotion
v-vv 

# ╔═╡ a5268ef8-425e-4b8a-b62a-44af4a4993b7
# This is again a vector
w=rand(5)

# ╔═╡ 24dcfabb-b9a1-4c60-afe2-2b29ab653b6d
# First column is a 5 x 1 matrix, second column is a vector of length 5
Mv=[v w]

# ╔═╡ 59bebc62-f680-4d08-9ed9-febe8d8e58d6
# # Matrix columns and rows are extracted as vectors
x=Mv[:,1] 

# ╔═╡ 270b5108-2f91-497e-9f01-187c8c3f4d6a
y=Mv[:,2]

# ╔═╡ eb2954a3-b774-45bb-9776-8015daed12e1
x==v # The types differ

# ╔═╡ 7ad96bf2-08c4-43e7-8d79-5aecc90ff9d8
y==w

# ╔═╡ f56b5ed0-7c29-11eb-2b4f-3d73065ae542
# Row is also a vector
Mv[1,:]

# ╔═╡ 1d361187-3759-41a0-9760-a234bfc054f1
# Transpose of a matrix is adjoint
v'

# ╔═╡ 0dd59a04-f872-4c56-8a99-d586086b5802
# Transpose of a vector is adjoint
w'

# ╔═╡ 5cb26aec-503d-4b62-87aa-6d695a2e7c31
md"""
## 1D and 2D arrays
"""

# ╔═╡ c5552f19-1f48-4910-bf92-ef4d1816fcba
# Range
w₁=1.0:5

# ╔═╡ 29d32710-3f61-4dad-b1a2-41cca022de6c
A*w₁

# ╔═╡ 5a89699f-e40f-49a0-ae31-561466963ce3
# Vector
w₂=collect(1.0:5) 

# ╔═╡ 19c4f76b-a9f2-4c78-b0a3-0f2bf3109b6f
A*w₂

# ╔═╡ d31925e5-5c02-4386-8181-d308b3f937f4
# This returns a 2-dimensional array - v is a 5 x 1 array
A*v

# ╔═╡ c919d9d4-079e-490a-b858-762bac13c648
# Usual matrix multiplication formula does not work. Why?
B₁=[A[i,:]*A[:,j] for i=1:5, j=1:5]

# ╔═╡ f4ba5280-7cd2-11eb-1cb5-e3bf3534f077
A

# ╔═╡ b5263af0-bd84-4086-9481-eaf9e8b6d402
# Rows and columns are both 1D vectors - must use dot product!
B₂=[A[i,:]⋅A[:,j] for i=1:5, j=1:5]

# ╔═╡ 19abf655-4f37-44aa-9557-e776fb9e5392
inv(B₂)

# ╔═╡ 30e36ce9-896c-4714-b82c-17736ed6b508
md"""
# `ones()`, `one()`, `Uniform Scaling()`, `zero()`, and `zeros()`
   
Notice that the output type depends on the argument. This is a general Julia feature called `Multiple dispatch` 
and will be explained later in more detail. 
"""

# ╔═╡ 91294cdb-1bd9-4e36-8244-19ed3f3d9101
# The output type depends on the argument. Float64 is the default.
ones(3,5), ones(Int,5)

# ╔═╡ 7b86d9a0-7c38-11eb-3ca5-df8e517f3a99
zeros(ComplexF64,3,5)

# ╔═╡ 6b28a250-7c38-11eb-200d-c7c0836084f0
zero(ComplexF64)

# ╔═╡ 99a7b5d0-7c38-11eb-291e-1deab770ea9a
one(Int)

# ╔═╡ d46bd8e2-0c27-4e76-98a8-85d73940ff92
zeros(3,5)

# ╔═╡ f3253ec4-3aaa-4b51-bded-4c37b17e6208
zeros(5)

# ╔═╡ a3ee2c40-7c38-11eb-0c7a-359758b3af8c
# I is automatically recognised
zeros(Int,3,3)+I

# ╔═╡ ae870870-7c38-11eb-1b1f-475e0119efa2
# Multiplicative identity
one(rand(3,3))

# ╔═╡ 1a6706f5-d464-4b4d-856a-c969aadd44e1
md"""
# Complex numbers

`i` is too valuable symbol for loops, so Julia uses `im` for the complex unit. 
"""

# ╔═╡ 72faab81-f1ce-4d66-9b4d-348abaecd87d
im

# ╔═╡ 8a6ddee3-9f38-4074-83cf-64e919d2ba2e
2im

# ╔═╡ 8d62321c-985e-4399-9ea7-d89c04b430ac
typeof(2.0im)

# ╔═╡ 809b89d6-7868-4ca3-8012-ffd10aca0cd7
# Another way of defining complex numbers
complex(3,4)

# ╔═╡ c5b96be9-2b53-4cbc-aad4-2b0ef59b84df
# Notice promotion
complex(3,4.0)

# ╔═╡ e71d851f-89e9-4c4a-9e41-0c4f0a517013
# This produces an error (like in any other language)
sqrt(-1) 

# ╔═╡ 13761e35-aeff-4463-8a3a-3be47262b889
# and this is fine.
sqrt(complex(-1))

# ╔═╡ 37c50436-f17c-4cbe-a8cd-41cc48555596
md"""
# Ternary operator
Let us define our version of the sign function
"""

# ╔═╡ 8c7c1862-4a10-498d-a4aa-bcd0ac00ad62
si(x) = (x>0) ? 1 : -1

# ╔═╡ 7f606319-7cbc-4338-9fcd-c00a5f6a32f0
si(-13)

# ╔═╡ d35fefc3-a529-4021-a91a-08bc5bf80ebb
md"""
This is equivalent to:
"""

# ╔═╡ d986ebe2-76ec-42bf-8c6e-d486841fbe36
function si₁(x)
    if x>0
        return 1
    else
        return -1
    end
end

# ╔═╡ 634626ed-739d-4900-b639-b1447ec9ee43
si(pi-8), si(0), si₁(0.0)

# ╔═╡ f9968b8a-6474-4f1c-b65c-e0bc2187a70e
md"""
The expressions can be nested:
"""

# ╔═╡ 3a1e5e6a-6235-498c-a3e4-c19e4536d63d
# now si(0) is 0
si₂(x) = (x>0) ? 1 : ((x<0) ? -1 : 0)

# ╔═╡ 1a842df8-274a-4ed9-afa7-c1c7430f9283
si₂(π-8), si₂(0)

# ╔═╡ 28455bbd-5bfc-4f98-be4f-c26311f7e327
md"""
# Typing
Special mathematical (LaTeX) symbols can be used (like $\alpha$, $\Xi$, $\pi$, $\oplus$, $\cdot$, etc.). The symbol in both, the notebook and command line version, is produced by writing 
LaTeX command followed by `<Tab>`.
    
Subscripts and superscripts are written as e.g., x\\_m`<TAB>`, x\^3`<TAB>`.

Julia uses full UNICODE set as variable names.
"""

# ╔═╡ 312bff26-67c8-4b5c-b61f-b8d66ece718a
Ξ = 8; Ψ  = 6; Γ = Ξ ⋅ Ψ; 🐮=√2

# ╔═╡ f0c39851-7ac9-4959-a48e-2612211de173
typeof(Γ)

# ╔═╡ a1b56d48-45c2-4483-81bb-f5829ff10e7e
begin
	ω₁=7; xᵏ=23
	ω₁*xᵏ
end

# ╔═╡ 6394216f-fcae-4dc8-8420-432d0b273621
md"""
# Writing a program and running a file

In Julia, the output of command is not displayed, unless explicitely required. 
Only the output of the last command is given. 

To display results you can use commands `@show` or `println()`.

Consider the file `deploy.jl` with the following code
```julia
n=map(Int,ARGS[1])          # take one integer argument
println(rand(1:n,n,n))  # generate and print n x n matrix of random integers between 1 and n
@show b=3               # set b to 3 and show the result
c=4                     # set c to 4
```
Running the program in the unix shell gives
```julia
$ julia deploy.jl 5
[1 3 2 4 1
 5 3 1 1 4
 5 4 2 2 5
 3 1 2 3 4
 4 4 5 4 4]
b = 3 => 3
```
Notice that the result of the last command (_c_) is not displayed.


Similarly, the program can be converted to executable and run directly, 
without referencing `julia` in the command line.
The refernece to `julia` must be added in the first line, as in the file `deploy1.jl`:
```julia
#!/usr/bin/julia
n=map(Int,ARGS[1])
println(rand(1:n,n,n))
@show b=3
c=4
```
In the shell do:
```
$ chmod +x deploy1.jl
$ ./deploy1.jl 5
[4 5 3 2 5
 4 2 1 5 1
 3 2 4 5 1
 2 4 4 3 1
 3 4 5 3 3]
b = 3 => 3
```

Finally, to run the same program in `julia` shell or `IJulia`, the input has to be changed, as in the file `deploy2.jl`:
```julia
n=parse(Int,readline())
println(rand(1:n,n,n))
@show b=3
c=4
```
__Notice that now the result of the last line is displayed by default__  - in this case it is `4`, the values of `c`. The output of the random matrix and of `b` is forced.


Input in Pluto notebook must be handled differently:
"""

# ╔═╡ 505f8a60-7c3d-11eb-1ce1-c9b7cf6e6c6a
@bind s TextField()

# ╔═╡ 34d0a3ae-7c3d-11eb-0d94-35472ded534e
n = (s=="" ? "" : parse(Int,s))

# ╔═╡ 97de847a-2457-47f1-96a2-6f1214e81a5e
md"""
# Running external programs and unix pipe

## `run()` - calling external program
"""

# ╔═╡ 67d1137f-146b-49bd-94cb-cc666102cb15
md"""
Notice, that this is not a gret help, Julia has much better commands for this.
"""

# ╔═╡ 6f06c3c6-53e2-4374-aeaa-fbcad7db7204
# This calls the unix Calendar program
# run(`cal`)

# ╔═╡ fe3af7f1-bde9-4a27-99b6-665386b38af9
# The pipe is '|>' instead of usual '|'
# run(pipeline(`cal`,`grep Sa`)) 

# ╔═╡ 081ccb66-0b04-41b1-bae5-9d3e1f9b5c69
md"""
## `ccall()` - calling `C` program
"""

# ╔═╡ 37e4a4d1-99a7-49ff-8daf-6b204aa4a3b5
# Simple version
ccall(:clock,Int,()) 

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
PlutoUI = "~0.7.10"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[HypertextLiteral]]
git-tree-sha1 = "72053798e1be56026b81d4e2682dbe58922e5ec9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.0"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "438d35d2d95ae2c5e8780b330592b6de8494e779"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.0.3"

[[PlutoUI]]
deps = ["Base64", "Dates", "HypertextLiteral", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "Suppressor"]
git-tree-sha1 = "26b4d16873562469a0a1e6ae41d90dec9e51286d"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.10"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[Suppressor]]
git-tree-sha1 = "a819d77f31f83e5792a76081eee1ea6342ab8787"
uuid = "fd094767-a336-5f1f-9728-57cf17d0bbfb"
version = "0.2.0"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
"""

# ╔═╡ Cell order:
# ╟─1f472327-7696-4070-877b-36fa00b32d11
# ╠═1685eaa0-7c38-11eb-0dc2-8bfbea75d947
# ╟─5d1910ad-80e4-4ea0-972f-3cb47ae7f057
# ╟─3275dbbd-8612-4395-adb9-1318cc85d136
# ╟─8f8e9dba-72ba-41e1-b39a-aeef95402555
# ╟─d9196675-4ec2-4236-a7f1-bcc589e234ab
# ╠═4f2aee7c-6e08-404c-a561-18f1662e5837
# ╟─b46be41b-edc5-4321-850c-ecb82e13ac7a
# ╠═dbc23df0-c4f0-4efc-a452-3392e0a18bf1
# ╟─9779cbcd-d500-4383-a7e3-ff909837aaa6
# ╟─2b973dd0-1e52-44c9-bf22-be092080fa3e
# ╠═334ea76b-a126-41b0-8945-ba166507b0a1
# ╠═9ee1a335-8060-4c6c-b1c6-4e1699053086
# ╠═9293e691-1b04-4989-a891-fbd1bd3306fe
# ╟─2d16ab49-9244-4ef6-b5e0-fa5304b367e4
# ╠═9e98e5ce-3ca2-4bf8-84f1-23dc3e793b9e
# ╟─6acd3389-37b1-48a7-9a04-795372503bac
# ╠═2078e00d-b3b2-4188-b5be-9835de2ab357
# ╠═c75546c0-47ae-4706-9a73-aad97700584d
# ╠═0bc90ce5-9976-498d-930c-e34fea9c7193
# ╠═3d949c09-801b-434d-a972-f590e8de8704
# ╠═963ea673-6469-4718-b348-a281578c7742
# ╠═ae3fb606-c31d-4edd-9d4c-4c3982c86be4
# ╠═e4790db9-99ee-4200-bc30-d80d0494f23f
# ╠═9efef8f4-130b-4286-b109-a43e47a1d00b
# ╠═b80eb1ee-0d74-47fe-8fbf-87140fad8e7c
# ╠═de3245f1-8b93-4e9e-be8c-76f7f096907f
# ╠═6cc8eb48-4dbb-4599-b126-b4b89f8376de
# ╠═cdff612c-ad32-4fc8-aa11-d4501f490f05
# ╠═4725ae25-0800-442e-88b5-19243da2a3e9
# ╠═48da687f-935d-4262-ad0a-f5e8f334da9a
# ╠═0a918be5-070d-4f41-a09d-dc5d3238189b
# ╟─84e88328-de83-49ed-89f8-f9ba13f1ea52
# ╠═a232e8d5-1a28-4a85-9cd4-e58bb79b77a0
# ╠═851065e9-d701-46ae-9f22-d6ef0975694f
# ╠═029ac818-a082-4454-98fc-b4a9b94b719a
# ╠═af54e7ef-3512-4cdf-bea5-e7848299638d
# ╠═a5268ef8-425e-4b8a-b62a-44af4a4993b7
# ╠═24dcfabb-b9a1-4c60-afe2-2b29ab653b6d
# ╠═59bebc62-f680-4d08-9ed9-febe8d8e58d6
# ╠═270b5108-2f91-497e-9f01-187c8c3f4d6a
# ╠═eb2954a3-b774-45bb-9776-8015daed12e1
# ╠═7ad96bf2-08c4-43e7-8d79-5aecc90ff9d8
# ╠═f56b5ed0-7c29-11eb-2b4f-3d73065ae542
# ╠═1d361187-3759-41a0-9760-a234bfc054f1
# ╠═0dd59a04-f872-4c56-8a99-d586086b5802
# ╠═5cb26aec-503d-4b62-87aa-6d695a2e7c31
# ╠═c5552f19-1f48-4910-bf92-ef4d1816fcba
# ╠═29d32710-3f61-4dad-b1a2-41cca022de6c
# ╠═5a89699f-e40f-49a0-ae31-561466963ce3
# ╠═19c4f76b-a9f2-4c78-b0a3-0f2bf3109b6f
# ╠═d31925e5-5c02-4386-8181-d308b3f937f4
# ╠═c919d9d4-079e-490a-b858-762bac13c648
# ╠═f4ba5280-7cd2-11eb-1cb5-e3bf3534f077
# ╠═b5263af0-bd84-4086-9481-eaf9e8b6d402
# ╠═19abf655-4f37-44aa-9557-e776fb9e5392
# ╟─30e36ce9-896c-4714-b82c-17736ed6b508
# ╠═91294cdb-1bd9-4e36-8244-19ed3f3d9101
# ╠═7b86d9a0-7c38-11eb-3ca5-df8e517f3a99
# ╠═6b28a250-7c38-11eb-200d-c7c0836084f0
# ╠═99a7b5d0-7c38-11eb-291e-1deab770ea9a
# ╠═d46bd8e2-0c27-4e76-98a8-85d73940ff92
# ╠═f3253ec4-3aaa-4b51-bded-4c37b17e6208
# ╠═a3ee2c40-7c38-11eb-0c7a-359758b3af8c
# ╠═ae870870-7c38-11eb-1b1f-475e0119efa2
# ╟─1a6706f5-d464-4b4d-856a-c969aadd44e1
# ╠═72faab81-f1ce-4d66-9b4d-348abaecd87d
# ╠═8a6ddee3-9f38-4074-83cf-64e919d2ba2e
# ╠═8d62321c-985e-4399-9ea7-d89c04b430ac
# ╠═809b89d6-7868-4ca3-8012-ffd10aca0cd7
# ╠═c5b96be9-2b53-4cbc-aad4-2b0ef59b84df
# ╠═e71d851f-89e9-4c4a-9e41-0c4f0a517013
# ╠═13761e35-aeff-4463-8a3a-3be47262b889
# ╟─37c50436-f17c-4cbe-a8cd-41cc48555596
# ╠═8c7c1862-4a10-498d-a4aa-bcd0ac00ad62
# ╠═7f606319-7cbc-4338-9fcd-c00a5f6a32f0
# ╟─d35fefc3-a529-4021-a91a-08bc5bf80ebb
# ╠═d986ebe2-76ec-42bf-8c6e-d486841fbe36
# ╠═634626ed-739d-4900-b639-b1447ec9ee43
# ╟─f9968b8a-6474-4f1c-b65c-e0bc2187a70e
# ╠═3a1e5e6a-6235-498c-a3e4-c19e4536d63d
# ╠═1a842df8-274a-4ed9-afa7-c1c7430f9283
# ╟─28455bbd-5bfc-4f98-be4f-c26311f7e327
# ╠═312bff26-67c8-4b5c-b61f-b8d66ece718a
# ╠═f0c39851-7ac9-4959-a48e-2612211de173
# ╠═a1b56d48-45c2-4483-81bb-f5829ff10e7e
# ╟─6394216f-fcae-4dc8-8420-432d0b273621
# ╠═505f8a60-7c3d-11eb-1ce1-c9b7cf6e6c6a
# ╠═34d0a3ae-7c3d-11eb-0d94-35472ded534e
# ╠═97de847a-2457-47f1-96a2-6f1214e81a5e
# ╟─67d1137f-146b-49bd-94cb-cc666102cb15
# ╠═6f06c3c6-53e2-4374-aeaa-fbcad7db7204
# ╠═fe3af7f1-bde9-4a27-99b6-665386b38af9
# ╟─081ccb66-0b04-41b1-bae5-9d3e1f9b5c69
# ╠═37e4a4d1-99a7-49ff-8daf-6b204aa4a3b5
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
