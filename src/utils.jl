"""
    @staticdef struct Struct
        foo
        bar::Typ
        ...
    end

Generate a 'static' version of a struct where each field is encoded as a type parameter.
This is useful when the field values should be specialized on, e.g., in the context of GPU
kernels.

The macro will generate a struct definition, including a constructor that takes each field
as an argument (in the same order, and using the same types as the original definition).
In addition, a `getproperty` accessor is defined such that the fields can be accessed
using convenient dot syntax (i.e., `obj.field`).
"""
macro staticdef(ex)
    # decode struct definition
    Meta.isexpr(ex, :struct) || error("@staticdef: expected struct definition")
    is_mutable, struct_name, struct_body = ex.args
    is_mutable && error("@staticdef: struct definition must be immutable")
    ## decode fields
    @assert Meta.isexpr(struct_body, :block)
    fields = struct_body.args
    filter!(field -> !isa(field, LineNumberNode), fields)
    field_names = Symbol[]
    field_types = Dict{Symbol, Any}()
    for field in fields
        if Meta.isexpr(field, :(::))
            name, typ = field.args
        else
            name = field
            typ = Any
        end
        push!(field_names, name)
        field_types[name] = typ
    end

    # generate new struct definition, forwarding args to typevars
    typevars = Symbol.(uppercase.(String.(field_names)))
    struct_ex = quote
        struct $(esc(struct_name)){$(typevars...)}
            function $(esc(struct_name))($(fields...))
                new{$(field_names...)}()
            end
        end
    end

    # generate a getproperty accessor
    getproperties_ex = quote end
    if !isempty(field_names)
      current = nothing
      for field_name in field_names
          typevar = Symbol(uppercase(String(field_name)))
          test = :(field === $(QuoteNode(field_name)))
          if current === nothing
              current = Expr(:if, test, typevar)
              getproperties_ex = current
          else
              new = Expr(:elseif, test, typevar)
              push!(current.args, new)
              current = new
          end
      end
      ## finally, call `getfield` to emit an error
      push!(current.args, :(getfield(conf, field)))
      getproperties_ex = quote
          function Base.getproperty(conf::$(esc(struct_name)){$(typevars...)},
                                    field::Symbol) where {$(typevars...)}
            $getproperties_ex
          end
      end
    end

    quote
        $struct_ex
        $getproperties_ex
    end
end
