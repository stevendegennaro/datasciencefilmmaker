### Adapted from https://github.com/davidsa03/numerize

def denumerize(n):
	try:
		sufixes = [ "", "K", "k", "M", "B", "T", "Qa", "Qu", "S", "Oc", "No", 
				   "D", "Ud", "Dd", "Td", "Qt", "Qi", "Se", "Od", "Nd","V", 
					"Uv", "Dv", "Tv", "Qv", "Qx", "Sx", "Ox", "Nx", "Tn", "Qa",
					"Qu", "S", "Oc", "No", "D", "Ud", "Dd", "Td", "Qt", "Qi",
					"Se", "Od", "Nd", "V", "Uv", "Dv", "Tv", "Qv", "Qx", "Sx",
					"Ox", "Nx", "Tn", "x", "xx", "xxx", "X", "XX", "XXX", "END"] 
	
		sci_expr = [1e0, 1e3, 1e3, 1e6, 1e9, 1e12, 1e15, 1e18, 1e21, 1e24, 1e27, 
					 1e30, 1e33, 1e36, 1e39, 1e42, 1e45, 1e48, 1e51, 1e54, 1e57, 
					 1e60, 1e63, 1e66, 1e69, 1e72, 1e75, 1e78, 1e81, 1e84, 1e87, 
					   1e90, 1e93, 1e96, 1e99, 1e102, 1e105, 1e108, 1e111, 1e114, 1e117, 
					  1e120, 1e123, 1e126, 1e129, 1e132, 1e135, 1e138, 1e141, 1e144, 1e147, 
					   1e150, 1e153, 1e156, 1e159, 1e162, 1e165, 1e168, 1e171, 1e174, 1e177]

		suffix_idx = len(n)

		for i in n:
			if i.isnumeric():
				continue
			elif i == '.':
				continue
			else:
				suffix_idx = n.index(i)
				break
			
		suffix = n[suffix_idx:]
		num = n[:suffix_idx]

		suffix_val = sci_expr[sufixes.index(suffix)]

		result = int(float(num) * suffix_val)

		return result

	except:
	    return 'err'
