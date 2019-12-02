
from __future__ import division

from HTMLParser import HTMLParser
from multiprocessing import Pool
import urllib, pickle, time, random

# class ArticleMarkdownHTMLParser(HTMLParser):

# 	def __init__(self):
# 		HTMLParser.__init__(self)
# 		self.focus = False
# 		self.text = ''
# 	def handle_starttag(self, tag, attrs):
# 		if tag=='textarea':
# 			self.focus = True
# 	def handle_endtag(self, tag):
# 		if tag=='textarea':
# 			self.focus = False
# 	def handle_data(self, data):
# 		if self.focus:
# 			self.text = self.text + data

def get_article_markdown(article):
	if random.random()<0.01:
		time.sleep(10)
	url = 'https://www.wikihow.com/index.php?title=%s&action=edit&advanced=true'%urllib.quote(article.replace(' ', '-'))
	opener = urllib.URLopener()
	opener.addheaders = [('User-Agent', 
		'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.94 Safari/537.36'), ('Accept', '*/*')]
	f = opener.open(url)
	content = f.read()
	# print content.split('<textarea')[1].split('>')[1].split('</textarea>')[0]
	from bs4 import BeautifulSoup
	soup = BeautifulSoup(content, from_encoding=f.info().getparam('charset'), features='html.parser')
	tags = soup.find_all('textarea')
	assert len(tags)==1 and tags[0]['id']=='wpTextbox1', article
	# tag = [t for t in tags if t['id']=='wpTextbox1' or t['id']=='steps_text']
	# assert len(tag) == 1
	# tag = tag[0]
	tag = tags[0]
	return tag.text.replace(u"\u2018", "'").replace(u"\u2019", "'")

if __name__ == '__main__':
	count = 0
	all_articles = []
	article_lists = pickle.load(open('article_lists.pkl', 'rb'))
	for cat, al in article_lists.iteritems():
		for a in al:
			all_articles.append(a)
	print len(all_articles)
	print len(set(all_articles))
	# article_mds = dict()
	# article_mds = pickle.load(open('article_mds_part2.pkl', 'rb'))
	# for a in all_articles:
	# 	print count, a
	# 	count += 1
	# 	if a in article_mds:
	# 		continue
	# 	try:
	# 		article_mds[a] = get_article_markdown(a)
	# 	except IOError:
	# 		pickle.dump(article_mds, open('article_mds_part3.pkl', 'wb'))
	# 		quit()
	# pickle.dump(article_mds, open('article_mds.pkl', 'wb'))
