
from __future__ import division

from HTMLParser import HTMLParser
import urllib, pickle

class MyHTMLParser(HTMLParser):

	def __init__(self):
		HTMLParser.__init__(self)
		self.num_divs = 0
		self.all_subcategories = []

	def handle_starttag(self, tag, attrs):
		if tag=='div' and ('id', 'cat_sub_categories') in attrs:
			self.num_divs += 1
			return
		if self.num_divs==0:
			return
		if tag=='div':
			self.num_divs += 1

	def handle_endtag(self, tag):
		if self.num_divs==0:
			return
		if tag=='div':
			self.num_divs -= 1

	def handle_data(self, data):
		if self.num_divs==0:
			return
		if data.strip()!='' and data.strip()!='Related Topics':
			self.all_subcategories.append(data)

	def get_main_subcategories(self):
		main = []
		dot = '\xc2\xb7'
		for i in xrange(len(self.all_subcategories)):
			if self.all_subcategories[i]==dot:
				continue
			if i==len(self.all_subcategories)-1 or self.all_subcategories[i+1]!=dot:
				main.append(self.all_subcategories[i])
		return main

def get_main_subcategories(cat):
	url = 'https://www.wikihow.com/Category:'+cat.replace(' ', '-')
	opener = urllib.URLopener({})
	f = opener.open(url)
	content = f.read()
	if '<h2>Related Topics</h2>' not in content:
		return []
	parser = MyHTMLParser()
	parser.feed(content)
	main = parser.get_main_subcategories()
	if cat in main:
		main.remove(cat)
	return main

def get_tree(cat):
	return get_tree_helper(cat, [('ROOT', cat)])

def get_tree_helper(parent_cat, cur_list):
	print parent_cat
	child_cats = get_main_subcategories(parent_cat)
	if child_cats==[]:
		return cur_list
	for cat in child_cats:
		cur_list.append((parent_cat, cat))
	for cat in child_cats:
		get_tree_helper(cat, cur_list)
	return cur_list

cat_tree = get_tree('Home and Garden')
pickle.dump(cat_tree, open('category_tree.pkl', 'wb'))
print len(cat_tree)
