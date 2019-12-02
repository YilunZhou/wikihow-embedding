
from __future__ import division

from HTMLParser import HTMLParser
import urllib, pickle

class NormalArticleHTMLParser(HTMLParser):

	def __init__(self):
		HTMLParser.__init__(self)
		self.in_span = False
		self.in_article_sec = False
		self.articles = []
		self.div_count = 0
		self.cur_texts = []

	def handle_starttag(self, tag, attrs):
		if ('id', 'cat_all') in attrs:
			self.in_article_sec = True
			self.div_count += 1
		elif tag=='div' and self.in_article_sec:
			self.div_count += 1
		if tag=='span':
			self.in_span = True
		else:
			assert not self.in_span

	def handle_endtag(self, tag):
		if tag=='div' and self.in_article_sec:
			self.div_count -= 1
			if self.div_count==0:
				self.in_article_sec = False
		if tag=='span':
			self.in_span = False
			if self.in_article_sec:
				self.articles.append(str(''.join(self.cur_texts)))
				self.cur_texts = []

	def handle_data(self, data):
		if self.in_span and self.in_article_sec:
			# self.articles.append(data)
			self.cur_texts.append(data)

	def handle_charref(self, ref):
		self.handle_entityref("#" + ref)

	def handle_entityref(self, ref):
		self.handle_data(self.unescape("&%s;" % ref))

def get_normal_articles(cat):
	url = 'https://www.wikihow.com/Category:'+cat.replace(' ', '-')
	opener = urllib.URLopener()
	opener.addheaders = [('User-Agent', 
		'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.94 Safari/537.36'), ('Accept', '*/*')]
	f = opener.open(url)
	content = f.read()
	parser = NormalArticleHTMLParser()
	parser.feed(content)
	return parser.articles

article_lists = dict()
categories = pickle.load(open('category_tree.pkl', 'rb'))
for parent, child in categories:
	if child not in article_lists:
		article_lists[child] = get_normal_articles(child)
		print child, len(article_lists[child])
pickle.dump(article_lists, open('article_lists.pkl', 'wb'))
