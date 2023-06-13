### A Pluto.jl notebook ###
# v0.19.20

using Markdown
using InteractiveUtils

# ╔═╡ 21b85823-6493-4551-bc31-e1811d11c910
begin
	import Pkg
	Pkg.activate(mktempdir())
	Pkg.add(Pkg.PackageSpec(url="https://github.com/ivanslapnicar/IncrementalSVD.jl"))
	using IncrementalSVD, PlutoUI
end

# ╔═╡ c9c5a3f1-1d3b-48d6-85e6-fd7de59d5a3f
md"""
# Solutions 4 - Age of Recommendation

After __Age of Search__ (see the Page Rank notebook), we are now in the 
__Age of Recommendation__. This notebook is about Netflix Recommendations using 
Simon Funk's algorithm as implemented in [IncrementalSVD.jl](https://github.com/aaw/IncrementalSVD.jl) by Aaron Windsor.
"""

# ╔═╡ 634b73c1-5cd7-48f9-a3c0-a96678f7857a
md"""
## Messages

Easy -> hard?

Present: __age of search__ -> _mathematics_

Future: __age of recommendation__  -> _mathematics_

(BigData, new technologies)

"""

# ╔═╡ b5e97ee7-7724-400d-8b8a-7dfb4002b482
md"""
## Easy -> hard

weight:  1 _kg_  -> 1 _kg_ $\pm$ 0.000000001 _kg_

running: 100 *m* -> 42,195 *m* ili 100 *m* < 10 *sek*

mathematics: exam -> state competition  -> [Olympiad](http://www.imo-official.org/)

search, recommending: good -> excellent
"""

# ╔═╡ 79487c86-7607-4c0e-bbc6-5b7e18d4629f
md"""
## Age of Search

google (and others)


* [50 billion pages](http://www.worldwidewebsize.com/), [3.5 billion querries daily](http://www.internetlivestats.com/google-search-statistics/)
* __PageRank__
* history, context - cookies, storing data (about you), [200+ parameters](http://backlinko.com/google-ranking-factors)
"""

# ╔═╡ 948f0427-b8c3-4d01-a7e6-114a3d13fa17
md"""
## Age of Recommendation

NetFlix, Amazon Prime, PickBox, ... - on-line streaming of movies and shows

[NetFlix](https://www.netflix.com/hr/)

 * [182 million users](https://www.statista.com/statistics/250934/quarterly-number-of-netflix-streaming-subscribers-worldwide/), 5,000 movies
 * [NetFlix Prize](http://www.kdd.org/kdd2014/tutorials/KDD%20-%20The%20Recommender%20Problem%20Revisited.pdf)
 
"""

# ╔═╡ 783abd90-cee5-4b59-ba15-3212987d3011
md"""
## Mathematics

Netflix Recommendation Engine is based on approximation of a (large and sparse) matrix
```
M = Users x Movies 
```
using (approximation of) singular value decomposition (SVD): 

* [IncrementalSVD.jl](https://github.com/aaw/IncrementalSVD.jl)
* [A parallel recommendation engine in Julia](http://juliacomputing.com/blog/2016/04/22/a-parallel-recommendation-engine-in-julia.html)
"""

# ╔═╡ e2973b1e-6333-4bf0-b567-654ab35bae26
md"""
## Similarities

Similarity of users $i$ and $k$:

$$
\cos \angle (i,k)=\frac{(M[i,:],M[k,:])}{\|M[i,:]\| \cdot \|M[k,:]\|}$$

Similarity of movies $i$ and $k$:

$$
\cos \angle (i,k)=\frac{(M[:,i],M[:,k)}{\|M[:,i]\| \cdot \|M[:,k]\|}$$
"""

# ╔═╡ 5b787963-e734-4f9d-b224-dbf6360fbcbd
md"""
## Search

Row $M[u,:]$ - what user $u$ thinks about movies

Column $M[:,m]$ - what users think about movie $m$

Element $M[u,m]$ - what user $u$ thinks about movie $m$.
"""

# ╔═╡ c512c71e-ea26-4ccd-9029-cdbd7fef07d2
md"""
## Problem

Matrix $M$ is sparse so we do not have enough information. For example, 

```
900188 marks / ( 6040 users x 3706 movies ) = 4%
```
"""

# ╔═╡ aabe1fa7-bec8-4ab9-97e2-308ebf491931
md"""
## Approximation

SVD decomposition $M=U\Sigma V^T$ is [approximated by a low-rank matrix](https://en.wikipedia.org/wiki/Low-rank_approximation) (for example, $\operatorname{rank}=25$)

 $(PlutoUI.LocalResource("./svd.png")) 



The approximation matrix is __full__ and __gives enough good information__.

Prize for an efficient approximation algorithm was USD $1.000.000$.  
"""

# ╔═╡ e2386262-d1b1-44ce-8ded-5e14e9bd1a0b


# ╔═╡ cf7044dd-3180-46e8-925d-81369d8261b5
varinfo(IncrementalSVD)

# ╔═╡ bb4fa13c-2504-44d6-b7e3-72e67f3fbd05
rating_set = load_small_movielens_dataset()

# ╔═╡ 4b31e1cd-d412-4c8e-99d0-59dab6a3f9e4
propertynames(rating_set)

# ╔═╡ f40046a7-5693-4ceb-8101-8553af6691df
# The format is (user, movie, mark)
rating_set.training_set

# ╔═╡ c5972e18-8b38-4d11-8f1c-a389524996f6
rating_set.test_set

# ╔═╡ 0ccd9ff0-ec14-46f3-9381-d6b98f0edae2
# Users and their IDs
rating_set.user_to_index

# ╔═╡ 10889093-768f-45ae-9c06-883c77fafb43
# Movies and their IDs
rating_set.item_to_index

# ╔═╡ 42f900e6-5676-45ea-a4c4-3ae0fd9be7e9
# We can extract the titles ...
keys(rating_set.item_to_index)

# ╔═╡ 5ae44ac7-e2dc-499e-aebb-9eeaff77ec2b
# or codes
values(rating_set.item_to_index)

# ╔═╡ b36b96ea-3a54-4375-8b89-aeda4701b538
# Which movies did the user "3000" grade?
user_ratings(rating_set, "3000")

# ╔═╡ 0824316c-f84a-4839-b5b0-f0f94d81ee02
# Let us find the exact title and code for "Blade Runner"
for k in keys(rating_set.item_to_index)
    if occursin("Blade",k)
        println(k)
    end
end

# ╔═╡ 197c08ee-7a72-41ab-a0ee-63e0461b4f7a
# Did the user "3000" grade "Blade Runner" ?
for k in user_ratings(rating_set,"3000")
    if occursin("Blade",k[1][2])
        println(k)
    end
end

# ╔═╡ ed5fff24-4d7b-491b-9d7c-c9128c24d95e
# How did the user "3000" grade "Sling Blade" ?
for k in user_ratings(rating_set,"3000")
    if occursin("Blade",k[1][2])
        println(k)
    end
end

# ╔═╡ 3a0c7828-c431-4264-a487-590abc945edb
get(rating_set.item_to_index,"Sling Blade (1996)",0)

# ╔═╡ 78d5e42f-ddf3-4218-9383-8b3bdbb103c9
get(rating_set.item_to_index,"Blade Runner (1982)",0)

# ╔═╡ 7c839d8f-9b30-4a5e-ae0e-d8db7e3ce30e
# This takes about half a minute
model = train(rating_set, 10);

# ╔═╡ bfabfbaf-10a3-4f21-8e3e-ef5bdfa9fec5
propertynames(model)

# ╔═╡ 72ba7a54-d130-41ae-97f7-41785bdcc971
model.U

# ╔═╡ bc0ae561-ff4d-44ab-9f0e-10426d6d1b50
model.S

# ╔═╡ ac1be542-d7d1-458f-a69c-2265e43798d4
model.V

# ╔═╡ 8ed49601-05ec-46a3-acf9-4d89269806c7
similar_items(model, "Friday the 13th (1980)",max_results=20)

# ╔═╡ f73609a5-2307-459f-ae2f-a9723e6a3bcf
# Take a look at the function
@which similar_items(model, "Friday the 13th (1980)")

# ╔═╡ 43153560-3c19-4091-85c8-15d8355ee6fa
similar_items(model, "Citizen Kane (1941)")

# ╔═╡ b068e6ed-88f5-46d7-b702-4ad44a97251e
similar_users(model,"3000",max_results=20)

# ╔═╡ 214e2d51-9e0f-46a1-a8cc-d02442d6ff2a
# What is the opinion of user "3000" about "Blade Runner (1982)" 
# in the approximate model (no true mark) ?
get_predicted_rating(model, "3000", "Blade Runner (1982)")

# ╔═╡ 5c3c4736-04cb-4d00-979d-10811059087f
# What is the opinion of user "3000" about "Citizen Kane (1941)"
# (no true mark!) ?
IncrementalSVD.get_predicted_rating(model, "3000", "Citizen Kane (1941)")

# ╔═╡ bf49ea10-2110-4f1e-ae33-8e734ddc34ac
# What is the opinion of user "3000" about "Sling Blade (1996)"
# in the approximate model (true mark 5.0) ?
IncrementalSVD.get_predicted_rating(model, "3000", "Sling Blade (1996)")

# ╔═╡ 2a12c12c-7f4a-478b-9c4a-393937613e00
# What is the opinion of user "3000" about "Time to Kill, A (1996)")
# in the approximate model (true mark 1.0) ?
IncrementalSVD.get_predicted_rating(model, "3000", "Time to Kill, A (1996)")

# ╔═╡ 1bd322b5-4c7b-434b-92f2-b729c19d34cd
md"""
## Thank you for you attention

### Questions?
"""

# ╔═╡ 635a02ec-0371-4fb1-a450-978d14dfc86b


# ╔═╡ Cell order:
# ╠═21b85823-6493-4551-bc31-e1811d11c910
# ╟─c9c5a3f1-1d3b-48d6-85e6-fd7de59d5a3f
# ╟─634b73c1-5cd7-48f9-a3c0-a96678f7857a
# ╟─b5e97ee7-7724-400d-8b8a-7dfb4002b482
# ╟─79487c86-7607-4c0e-bbc6-5b7e18d4629f
# ╟─948f0427-b8c3-4d01-a7e6-114a3d13fa17
# ╟─783abd90-cee5-4b59-ba15-3212987d3011
# ╟─e2973b1e-6333-4bf0-b567-654ab35bae26
# ╟─5b787963-e734-4f9d-b224-dbf6360fbcbd
# ╟─c512c71e-ea26-4ccd-9029-cdbd7fef07d2
# ╟─aabe1fa7-bec8-4ab9-97e2-308ebf491931
# ╠═e2386262-d1b1-44ce-8ded-5e14e9bd1a0b
# ╠═cf7044dd-3180-46e8-925d-81369d8261b5
# ╠═bb4fa13c-2504-44d6-b7e3-72e67f3fbd05
# ╠═4b31e1cd-d412-4c8e-99d0-59dab6a3f9e4
# ╠═f40046a7-5693-4ceb-8101-8553af6691df
# ╠═c5972e18-8b38-4d11-8f1c-a389524996f6
# ╠═0ccd9ff0-ec14-46f3-9381-d6b98f0edae2
# ╠═10889093-768f-45ae-9c06-883c77fafb43
# ╠═42f900e6-5676-45ea-a4c4-3ae0fd9be7e9
# ╠═5ae44ac7-e2dc-499e-aebb-9eeaff77ec2b
# ╠═b36b96ea-3a54-4375-8b89-aeda4701b538
# ╠═0824316c-f84a-4839-b5b0-f0f94d81ee02
# ╠═197c08ee-7a72-41ab-a0ee-63e0461b4f7a
# ╠═ed5fff24-4d7b-491b-9d7c-c9128c24d95e
# ╠═3a0c7828-c431-4264-a487-590abc945edb
# ╠═78d5e42f-ddf3-4218-9383-8b3bdbb103c9
# ╠═7c839d8f-9b30-4a5e-ae0e-d8db7e3ce30e
# ╠═bfabfbaf-10a3-4f21-8e3e-ef5bdfa9fec5
# ╠═72ba7a54-d130-41ae-97f7-41785bdcc971
# ╠═bc0ae561-ff4d-44ab-9f0e-10426d6d1b50
# ╠═ac1be542-d7d1-458f-a69c-2265e43798d4
# ╠═8ed49601-05ec-46a3-acf9-4d89269806c7
# ╠═f73609a5-2307-459f-ae2f-a9723e6a3bcf
# ╠═43153560-3c19-4091-85c8-15d8355ee6fa
# ╠═b068e6ed-88f5-46d7-b702-4ad44a97251e
# ╠═214e2d51-9e0f-46a1-a8cc-d02442d6ff2a
# ╠═5c3c4736-04cb-4d00-979d-10811059087f
# ╠═bf49ea10-2110-4f1e-ae33-8e734ddc34ac
# ╠═2a12c12c-7f4a-478b-9c4a-393937613e00
# ╟─1bd322b5-4c7b-434b-92f2-b729c19d34cd
# ╠═635a02ec-0371-4fb1-a450-978d14dfc86b
