import os
import numpy as np

input_template = """
tolerance 2.0 
output {output_name}.pdb 
filetype pdb 
structure lj.pdb 
  number {number}
  above plane 1 0 0 10
end structure 
"""

density = 0.5

with open('gen.input', 'w') as f:
    f.write(input_template.format(output_name=f'plane', number=0.5*20*20*20))

os.system('packmol < gen.input')
