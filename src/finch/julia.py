import os

#To change the version of Finch used, see the documentation for pyjuliapkg here: https://github.com/JuliaPy/pyjuliapkg
#Use pyjuliapkg to modify the `juliapkg.json` file in the root of this repo.
#An example development json is found in `juliapkg_dev.json`
import juliacall as jc  # noqa
from juliacall import Main as jl  # noqa

jl.seval("using Finch")
jl.seval("using Random")
