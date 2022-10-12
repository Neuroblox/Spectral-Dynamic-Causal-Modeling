using SparseDiffTools, ForwardDiff
using ForwardDiff: Partials, Dual
tagtype(::Dual{T,V,N}) where {T,V,N} = T
ForwardDiff.can_dual(::Type{ComplexF64}) = true

function f(x)
    y = [x[1] + im, x[2] + x[1] * im]

    if eltype(x) <: ForwardDiff.Dual
        y_vals = Complex.((p->p.value).(real(y)), (p->p.value).(imag(y)))
        y_part = (p->p.partials).(real(y)) + (p->p.partials).(imag(y))*im
        y = map((x1, x2) -> Dual{tagtype(x[1]), ComplexF64, length(x2)}(x1, Partials(Tuple(x2))), y_vals, y_part)
    end
    y
end
x = [3.0,2.0]
SparseDiffTools.forwarddiff_color_jacobian(f,x, ForwardColorJacCache(f, x, 2, tag = NeurobloxTag()), nothing)

bar = Dual{ForwardDiff.Tag{NeurobloxTag, ComplexF64}, ComplexF64, 12}(1.0 + im,ForwardDiff.Partials((1.0 + im,1.0 + im,1.0 + im,1.0 + im,1.0 + im,1.0 + im,1.0 + im,1.0 + im,1.0 + im,1.0 + im,1.0 + im,1.0 + im)))
