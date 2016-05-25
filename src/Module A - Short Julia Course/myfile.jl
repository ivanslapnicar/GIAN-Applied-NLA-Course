using Polynomials

function mypolyval(p::Poly,x::Number)
    s=p[0]
    t=one(x)
    for i=1:length(p)-1
        t*=x
        s+=p[i]*t
    end
    s
end 

function myhorner(p::Poly,x::Number)
    s=p[end]
    for i=length(p)-2:-1:0
        s=s*x+p[i]
    end
    s
end

n=1000001
pbig=Poly(rand(n));
x=0.12345

polyval(pbig,x)
mypolyval(pbig,x)
myhorner(pbig,x)

Profile.clear_malloc_data() 

polyval(pbig,x)
mypolyval(pbig,x)
myhorner(pbig,x)

exit
