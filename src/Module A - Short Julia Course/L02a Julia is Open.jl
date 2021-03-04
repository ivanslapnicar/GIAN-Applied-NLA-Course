### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ 55f8789f-5361-421f-9f37-98c154fd40cf
begin
	using Dates
	Dates.today()
end

# ╔═╡ 2db004b0-7c45-11eb-2c3a-3f0e111cf63a
using LinearAlgebra

# ╔═╡ 8ccdfcf9-d5d1-4b69-a854-15f2715bfafa
md"""
# Julia is Open - `varinfo()`, `methods()`, `@which`, ...


`Julia` is an open-source project, source being entirely hosted on [github](https://github.com/JuliaLang/julia).

The code consists of (these are old numbers, actual numbers may differ):

- 29K lines of `C/C++`
- 6K lines of `scheme`
- 68K lines of `julia`

Julia uses [LLVM](http://llvm.org) which itself has 680K lines of code. Therefore, `Julia` is very compact, compared to other languages, like LLVM's `C` compiler `clang` (513K lines of code) or `gcc`  (3,530K lines). This makes it easy to read the actuall code and get full information, in spite the fact that some parts of the documentation are insufficient. `Julia`'s "navigating" system,
consisting of commands `varinfo()`, `methods()` and `@which`, makes this even easier.

Further, the `Base` (core) of `Julia` is kept small, and the rest of the functionality is obtained through packages.
Since packages are written in `Julia`, they are navigated on the same way.




In this notebook, we demonstrate how to get help and navigate the source code.

## Prerequisites

Basic knowledge of programming in any language.

Read [Methods](https://docs.julialang.org/en/v1/manual/methods/#Methods) section of the `Julia` manual. (5 min)

## Competences 

The reader should be able to read the code and be able to find and understand calling sequences and outputs of any function.

## Credits 

Some examples are taken from [The Julia Manual](https://docs.julialang.org/en/v1/).

"""

# ╔═╡ 4440847c-db90-44a5-be48-08c5794ee5b7
md"""
## Operators `+`, `*` and `⋅`

Consider operators `+`, `*` and `⋅`, the first two of them seem rather basic in any language. The `⋅` symbol is typed as LaTeX command `\cdot` + `Tab`.

`?+` gives some information, which is vary sparse. We would expect more details, and we also suspect that `+` can be used in more ways that just hose two.

`?*` explaind more instances where `*` can be used, but the text itself is vague and not sufficient. 

`?⋅` appears to be what we expect fro the dot product off two vectors.
"""

# ╔═╡ 7a55f84e-1439-45ec-b428-5976cac084cd
#?+

# ╔═╡ 7514d88c-d0fa-45e5-947d-10fe0ce47f7d
md"""
## methods()

`Julia` functions have a feature called _multiple dispatch_, which means that the method depends on the name __AND__ the input.
Full range of existing methods for certain function name is given by the `methods()` command. 
> Running `methods(+)` sheds a completely differfent light on `+`.

The great Julia feature is that the links to the source code where the respective version of the function is defined, are readily provided. 
"""

# ╔═╡ 611770e1-4b5b-47a6-be3a-e0806990796d
md"""
### The `"+"` operator
"""

# ╔═╡ 9c208420-7cdf-11eb-2b99-e32e66b46e85
# methods(+)

# ╔═╡ bfb87460-7cdf-11eb-38ca-0961560c093b
# methodswith(Tridiagonal)

# ╔═╡ cb3e314e-68e0-4fc0-8ece-c5c8bc279f55
md"""
Following the third link, we get the following code snippet:
```
+(x::Bool) = int(x)
-(x::Bool) = -int(x)
+(x::Bool, y::Bool) = int(x) + int(y)
-(x::Bool, y::Bool) = int(x) - int(y)
*(x::Bool, y::Bool) = x & y
```

Therefore:
"""

# ╔═╡ 0f3c49cd-00c7-4654-9dc9-674a89e19a30
+(true), +(false),-(true),-(false)

# ╔═╡ 595878a2-30ab-4a19-a5ad-e621dd37f6c7
xb, yb = BitArray([0,1,0,1,0,1]), BitArray([0,1,1,1,1,0])

# ╔═╡ 2a255a09-3c3f-4fcb-99f2-b31adfb536b5
md"""
The above command is equivalent to 
```
xb = bitpack([0,1,0,1,0,1]); yb = bitpack([0,1,1,1,1,0])
```
except that only the last result would be displayed
"""

# ╔═╡ 4f6d3190-45c9-433d-b7f9-c42ed01dc04b
+xb,-(xb)

# ╔═╡ 60a3f371-d9b2-47a8-be8f-fab7cfb3fb48
xb+yb, +(xb,yb)

# ╔═╡ 74d98ae5-0641-4eea-9f8c-8e663a405f32
md"""
#### Manipulating dates

We see that one of the `+` methods is adding days to time: 

```
 +(x::Date,y::Base.Dates.Day) at dates/arithmetic.jl:77
```
Therefore, the 135-th day from today is:
"""

# ╔═╡ 48783355-31c8-4159-8dd5-8bea047c742c
dd=Dates.today()+Dates.Day(135)

# ╔═╡ c9a0df42-1368-4e5d-a21d-d924f7ffbb75
typeof(dd)

# ╔═╡ 79ca20e7-446e-4264-ae24-c46fbb4b15c1
md"""
More information about the two types can be obtained by `methods(Dates.Date)` and `methods(Dates.Day)`, respectively. 
"""

# ╔═╡ 7e6013ea-1e66-4a81-86a8-a69e22cde29a
md"""
#### Adding tridiagonal matrices

In the above output of `methods(+)`,  we see that we can add tridiagonal matrices:
```
+(A::Tridiagonal{T}, B::Tridiagonal{T}) at linalg/tridiag.jl:624
```
Following the link, we see that the method separately adds lower, main and upper diagonals, denoted by `dl`, `d` and `du`, respectively:
```
624: +(A::Tridiagonal, B::Tridiagonal) = Tridiagonal(A.dl+B.dl, A.d+B.d, A.du+B.du)
```
Let us see how exactly is the type `Tridiagonal` defined:
"""

# ╔═╡ 8591bb73-12e5-474e-b24b-7acfab2e3c5f
# methods(Tridiagonal)

# ╔═╡ 8c0b2316-26fa-4748-b047-3d2c6bf7acc0
md"""
This output seems confusing, but from the second line we conclude that we can define three diagonals, lower, main and upper diagonal, denoted as above. We also know that that the lower and upper diagonals are of size $n-1$. Let us try it out:
"""

# ╔═╡ 4163dd6e-7b5c-42f4-a9f4-c1e1a7faecba
T₁ = Tridiagonal(rand(6),rand(7),rand(6))

# ╔═╡ 89bf9619-7ee7-4ccd-99c0-48583e93bbe1
Matrix(T₁)

# ╔═╡ 684b3f4a-f83b-440d-9b13-d5dae159e77a
T₂ = Tridiagonal(rand(-5.0:5,6),randn(7),rand(-9.0:0,6))

# ╔═╡ 243854d4-495a-43de-88cb-1442275c8a8a
T₃=T₁+T₂

# ╔═╡ d79e7c32-ad00-4373-ae0e-04eeb8f6e333
md"""
This worked as expected, the result is again a `Tridiagonal`. We can access each diagonal by:
"""

# ╔═╡ 22af8627-6713-40a5-b716-81a55a6b7aad
# Output is in the console
println(T₃.dl, T₃.d, T₃.du)

# ╔═╡ 98142e4f-c703-4ea9-b893-5e2c831afa42
	md"""
### `@which`

Let us take a closer look at what happens. The `@which` command gives the link to the part of the code which is actually invoked. The argument should be only function, without assignment.
"""

# ╔═╡ dac4c688-cd5f-4ef5-8ba6-3709f945d3f0
@which T₁ + T₂

# ╔═╡ 60469aa5-8d5b-4996-a02d-100a65e0915f
@which Tridiagonal(rand(6),rand(7),rand(6))

# ╔═╡ c4bbf3a7-9497-474a-8174-db9530853486
md"""
In the code, we see that there is a type definition in the `struct` block:
```
## Tridiagonal matrices ##
struct Tridiagonal{T} <: AbstractMatrix{T}
    dl::Vector{T}    # sub-diagonal
    d::Vector{T}     # diagonal
    du::Vector{T}    # sup-diagonal
    du2::Vector{T}   # supsup-diagonal for pivoting
end
```
The `Tridiagonal` structure (or type) consists of __four__ vectors.
In our case, we actually called the function `Tridiagonal()` with __three__ vector arguments. The function creates the type of the same name, setting the fourth reqired vector `du2` to `zeros(T,n-2)`.

The next function with the same name is invoked when the input vectors have different types, in which case the types arer promoted to a most general one, if possible.
"""

# ╔═╡ 88ee9b5a-a1f1-49a1-b6cf-64ebac21634a
md"""
### `size()` and `Matrix()`

For each matrix type we need to define the function which returns the size of a matrix, and the function which converts the matrix of a given type to a full array. These function are listed after the second `Tridiagonal()` function.
"""

# ╔═╡ 5e2c532c-49f3-411c-b110-70c54263f007
T₄ = Tridiagonal([1,2,3],[2,3,4,5],[-1,1,2])

# ╔═╡ 7f026e3e-bf1c-49e5-b815-a9370eb5e3f5
size(T₄)

# ╔═╡ 20f20bfa-9aff-46a0-b358-5fe42640d21a
Matrix(T₄)	

# ╔═╡ bd0a3a98-4311-41b0-9209-5e09c4ecd9ad
md"""
### `sizeof()` 

Of course, using special types can leasd to much more efficient programs. For example, for `Tridiagonal` type, onlt four diagonals are stored, in comparison to storing full matrix when $n^2$ elements are stored. The storage used is obtained by the `sizeof()` function.
"""

# ╔═╡ e0a4eeaf-c259-453e-82aa-6813538ab536
T₁

# ╔═╡ 41daa15d-9b1f-47df-ad24-b124161ce25a
T₁f=Matrix(T₁)

# ╔═╡ cbb653fc-6ec1-408b-b683-7d6821724051
sizeof(T₁f)  #   392 =  7 * 7 * 8 bytes

# ╔═╡ e6a068a5-cb01-40f8-8f0c-19e84f8adbc0
# This is not yet implemented for Tridiagonal - only the storage 
# required for 4 vector variables' names is displayed
sizeof(T₁)

# ╔═╡ 7622b55d-49ba-4f21-8d85-4bfb95f0fe37
md"""
### `struct` is immutable

This means that we can change individual elements of defined parts, but not the parts as a whole (an alternative is to use the `type` construtor). For example: 
"""

# ╔═╡ 3e9cf5d9-f97f-468d-ba27-b6dc217418d8
begin
	@show T₄
	T₄.d[2]=123
	@show T₄
	# T₄.dl = [-1, -1 ,1]
end

# ╔═╡ ec7be6ff-07ff-4b6d-bf00-2c48a1535582
md"""
### `methodswith()`

This is the reverse of `methods()` - which methods exist for the given type. For example, what can we do with `Tridiagonal` matrices, or with `Dates.Day`:
"""

# ╔═╡ e54408ea-e984-42b8-926e-74806ffb5aec
# methodswith(Tridiagonal)

# ╔═╡ 017d94a3-c9cb-481f-b3ec-0dc687cb8a46
# methodswith(Dates.Day)

# ╔═╡ 9397e3da-f0b7-43da-9fd5-341a679e46c5
md"""
### The `"*"` operator
"""

# ╔═╡ 32050704-3017-4995-a054-5180f6e6e8d0
# methods(*)

# ╔═╡ 313b771e-95b0-4a6f-add9-e017ae6712a7
md"""
We can multiply various types of numbers and matrices. Notice, however, that there is no multiplication specifically defined for `Tridiagonal` matrices. This would not make much sense, since the product of two tridiagonal matrices is a pentadiagonal matrix, the product of three tridiagonal matrices is septadiagonal matrix, ...

Therefore, two tridiagonal matrices are first converted to full matrices, and then multiplied, as is seen in the source code.
"""

# ╔═╡ e1471fce-4bf3-43d7-8cfc-19e5d4f5b8ec
T₁*T₂

# ╔═╡ 3c8ae987-8cb5-43b4-bda8-bf528e3675af
@which T₁*T₂

# ╔═╡ a04fc04a-c446-4528-abfd-91be7edf0944
T₁*T₂*T₁

# ╔═╡ 8b29ea27-05ef-4b5e-8123-80667c385b70
md"""
### The "$\cdot$" operator
"""

# ╔═╡ 391b705d-0b2d-42b9-86a6-94416d2135cf
# methods(⋅)

# ╔═╡ 0b1557b6-2142-49db-8180-d7d17e1128cf
md"""
By inspecting the source, we see that the `scalar` or the `dot` product of two vectors (1-dimensional arrays) is 
computed via `BLAS` function `dot` for real arguments, and the function `dotc` for complex arguments.  
"""

# ╔═╡ 76edd9e2-e2c7-4048-b6a4-59f6ec93684a
begin
	x = rand(1:5,5)
	y  = rand(-5:0,5)
	z = rand(5)
	w = rand(5) + im*rand(5)
	@show x, y, z, w
	x⋅y, x⋅z, z⋅x, x⋅w, z⋅w, w⋅z
end

# ╔═╡ b48fd083-d317-4195-8500-c8fa5c5ac4ed
md"""
## `varinfo()`

The command `varinfo()` reveals the content of the specified package or module. It can be invoked either with the package name, or with the package name and a regular expression.
"""

# ╔═╡ 9f93e309-458e-45c8-9a20-dd2179b7b744
# varinfo(Dates)

# ╔═╡ 16ed9927-e764-434a-bbfb-f51fcd339a1f
# varinfo(LinearAlgebra)

# ╔═╡ 7e16a343-d2b6-44c0-9864-46703bb757e5
# Now with a regular expression - we are looking for 
# 'eigenvalue' related stuff. 
varinfo(LinearAlgebra, Regex("eig"))

# ╔═╡ c084e244-419a-46c1-8bb8-92f58324f7b1
md"""
Funally, let us list all we have in `Julia`'s `Base` module. __It is a long list!__ 
Notice that `Dates` and `LinearAlgebra` are modules themselves.
"""

# ╔═╡ 198883c4-3f6c-4725-a638-c3be9f7c0c12
# varinfo(Base)

# ╔═╡ a7de4d00-7c4a-11eb-1c76-cd159c6fdf9c
md"
## `code_llvm()` and `code_native()`

It is easy to see the LLVM or assebler code which is executed. The code is displayed in the console.
"

# ╔═╡ d399376e-7c4a-11eb-02e4-936801012749
f(a,b)=2a+b^2

# ╔═╡ e1aa988e-7c4a-11eb-2de6-7b7d5c2fc54c
@code_llvm f(1,2)

# ╔═╡ ef51c630-7c4a-11eb-3331-49602357198f
@code_native f(1,2)

# ╔═╡ 0521934e-7c4b-11eb-37f3-631715a07f3a
@code_native f(1.0,2)

# ╔═╡ 0ba1da00-7c4b-11eb-0abc-c395efc9700e
@code_native f(1,2+2.0im)

# ╔═╡ Cell order:
# ╟─8ccdfcf9-d5d1-4b69-a854-15f2715bfafa
# ╟─4440847c-db90-44a5-be48-08c5794ee5b7
# ╠═7a55f84e-1439-45ec-b428-5976cac084cd
# ╟─7514d88c-d0fa-45e5-947d-10fe0ce47f7d
# ╟─611770e1-4b5b-47a6-be3a-e0806990796d
# ╠═9c208420-7cdf-11eb-2b99-e32e66b46e85
# ╠═bfb87460-7cdf-11eb-38ca-0961560c093b
# ╟─cb3e314e-68e0-4fc0-8ece-c5c8bc279f55
# ╠═0f3c49cd-00c7-4654-9dc9-674a89e19a30
# ╠═595878a2-30ab-4a19-a5ad-e621dd37f6c7
# ╟─2a255a09-3c3f-4fcb-99f2-b31adfb536b5
# ╠═4f6d3190-45c9-433d-b7f9-c42ed01dc04b
# ╠═60a3f371-d9b2-47a8-be8f-fab7cfb3fb48
# ╟─74d98ae5-0641-4eea-9f8c-8e663a405f32
# ╠═55f8789f-5361-421f-9f37-98c154fd40cf
# ╠═48783355-31c8-4159-8dd5-8bea047c742c
# ╠═c9a0df42-1368-4e5d-a21d-d924f7ffbb75
# ╟─79ca20e7-446e-4264-ae24-c46fbb4b15c1
# ╟─7e6013ea-1e66-4a81-86a8-a69e22cde29a
# ╠═2db004b0-7c45-11eb-2c3a-3f0e111cf63a
# ╠═8591bb73-12e5-474e-b24b-7acfab2e3c5f
# ╟─8c0b2316-26fa-4748-b047-3d2c6bf7acc0
# ╠═4163dd6e-7b5c-42f4-a9f4-c1e1a7faecba
# ╠═89bf9619-7ee7-4ccd-99c0-48583e93bbe1
# ╠═684b3f4a-f83b-440d-9b13-d5dae159e77a
# ╠═243854d4-495a-43de-88cb-1442275c8a8a
# ╟─d79e7c32-ad00-4373-ae0e-04eeb8f6e333
# ╠═22af8627-6713-40a5-b716-81a55a6b7aad
# ╟─98142e4f-c703-4ea9-b893-5e2c831afa42
# ╠═dac4c688-cd5f-4ef5-8ba6-3709f945d3f0
# ╠═60469aa5-8d5b-4996-a02d-100a65e0915f
# ╟─c4bbf3a7-9497-474a-8174-db9530853486
# ╟─88ee9b5a-a1f1-49a1-b6cf-64ebac21634a
# ╠═5e2c532c-49f3-411c-b110-70c54263f007
# ╠═7f026e3e-bf1c-49e5-b815-a9370eb5e3f5
# ╠═20f20bfa-9aff-46a0-b358-5fe42640d21a
# ╟─bd0a3a98-4311-41b0-9209-5e09c4ecd9ad
# ╠═e0a4eeaf-c259-453e-82aa-6813538ab536
# ╠═41daa15d-9b1f-47df-ad24-b124161ce25a
# ╠═cbb653fc-6ec1-408b-b683-7d6821724051
# ╠═e6a068a5-cb01-40f8-8f0c-19e84f8adbc0
# ╟─7622b55d-49ba-4f21-8d85-4bfb95f0fe37
# ╠═3e9cf5d9-f97f-468d-ba27-b6dc217418d8
# ╟─ec7be6ff-07ff-4b6d-bf00-2c48a1535582
# ╠═e54408ea-e984-42b8-926e-74806ffb5aec
# ╠═017d94a3-c9cb-481f-b3ec-0dc687cb8a46
# ╟─9397e3da-f0b7-43da-9fd5-341a679e46c5
# ╠═32050704-3017-4995-a054-5180f6e6e8d0
# ╟─313b771e-95b0-4a6f-add9-e017ae6712a7
# ╠═e1471fce-4bf3-43d7-8cfc-19e5d4f5b8ec
# ╠═3c8ae987-8cb5-43b4-bda8-bf528e3675af
# ╠═a04fc04a-c446-4528-abfd-91be7edf0944
# ╟─8b29ea27-05ef-4b5e-8123-80667c385b70
# ╠═391b705d-0b2d-42b9-86a6-94416d2135cf
# ╟─0b1557b6-2142-49db-8180-d7d17e1128cf
# ╠═76edd9e2-e2c7-4048-b6a4-59f6ec93684a
# ╟─b48fd083-d317-4195-8500-c8fa5c5ac4ed
# ╠═9f93e309-458e-45c8-9a20-dd2179b7b744
# ╠═16ed9927-e764-434a-bbfb-f51fcd339a1f
# ╠═7e16a343-d2b6-44c0-9864-46703bb757e5
# ╟─c084e244-419a-46c1-8bb8-92f58324f7b1
# ╠═198883c4-3f6c-4725-a638-c3be9f7c0c12
# ╟─a7de4d00-7c4a-11eb-1c76-cd159c6fdf9c
# ╠═d399376e-7c4a-11eb-02e4-936801012749
# ╠═e1aa988e-7c4a-11eb-2de6-7b7d5c2fc54c
# ╠═ef51c630-7c4a-11eb-3331-49602357198f
# ╠═0521934e-7c4b-11eb-37f3-631715a07f3a
# ╠═0ba1da00-7c4b-11eb-0abc-c395efc9700e
