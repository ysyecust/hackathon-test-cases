import os
import numpy as np

input_template = """
tolerance 2.0 
output {output_name}.pdb 
filetype pdb 
structure lj.pdb 
  number {number}
  inside cube 0. 0. 0. {box_size}
end structure 
"""

density = 0.5

box_size = np.linspace(5, 25, 10)
number = box_size**3 * density

for s, n in zip(box_size.astype(int), number.astype(int)):

    with open('gen.input', 'w') as f:
        f.write(input_template.format(output_name=f's_{s}_n_{n}', number=n, box_size=s))

    os.system('../packmol-20.14.2/packmol < gen.input')