import numpy as np
from collections import namedtuple, defaultdict
from os import path

Section = namedtuple("Section", "timestep temperature potential_energy")

def load_data(filename):
	with open(path.join("data", filename), "r") as f:
		started = False
		section = None
		this_group = None
		groups = []
		sections = []

		for line in f:
			if line.startswith("TIMESTEP: 2000"):
				started = True
			if not started:
				continue
			if line.startswith("TIMESTEP:"):
				components = line.split("\t")
				section = Section(*(float(component.split(":")[1]) for component in components))
				sections.append(section)
				if this_group:
					groups.append(this_group)
				this_group = []
			else:
				this_group.append([float(n) for n in line.split(" ")])
		groups.append(this_group)

	return sections, np.array(groups)