
class Step:
	'''
	a Step object represent a single step, which has a short bold text and full original text
	'''
	def __init__(self, short, full):
		self.short = short
		self.full = full

class Subsection:
	'''
	a Subsection object represents a subsection which has a title and a list of Steps
	'''
	def __init__(self, title, steps, full):
		assert len(steps)!=0 and isinstance(steps[0], Step)
		self.title = title
		self.steps = steps
		self.full = full

class Article:
	'''
	an Article object represent an article. An article has a title, a full text, 
	either a list of steps or a list of subsections specified by format={'steps'|'subsections'}, 
	and sub_sec_type={'METHODS'|'STEPS'|'NA'} if format=='subsections', specifying whether
	subsections are different methods, different steps, or not specified in the article. 
	'''
	def __init__(self, title, full, format, data, sub_sec_type=None):
		assert format in ['steps', 'subsections']
		assert format=='steps' or sub_sec_type in ['METHODS', 'STEPS', 'NA']
		assert len(data)!=0 and (isinstance(data[0], Subsection) or isinstance(data[0], Step))
		self.title = title
		self.full = full
		self.format = format
		if format=='steps':
			self.steps = data
		else:
			self.subsections = data
			self.sub_sec_type = sub_sec_type
