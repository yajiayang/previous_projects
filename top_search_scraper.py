# Python2.7
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import string
import time
import codecs


class Belieber(object):
   def __init__(self):
       self.ff = webdriver.Firefox()
       self.ff.implicitly_wait(10)        
       self.ff.get("https://www.google.com")
       self.box = self.ff.find_element_by_id('lst-ib')
#       self.box = self.ff.find_element_by_class_name('input-wrapper')
       self.letters = string.ascii_lowercase

   def send_letters(self, term):
       with codecs.open('/Users/royyang/Desktop/Top_search_term/top_search_'+term+'.txt','w','utf-8') as new:
           for letter in self.letters:
               self.box.send_keys(term + ' ' + letter) 
#               self.box.send_keys(Keys.CONTROL, "a")
               time.sleep(0.3)
#               els = self.ff.find_elements_by_id('p_13838465-p')
               els = self.ff.find_elements_by_class_name('sbqs_c') #for google
#               els = self.ff.find_elements_by_class_name('sbsb_c gsfs')
               for elem in els:
                   new.write(u'{}\t{}\n'.format(term,elem.text))
#                   print term, elem.text, letter
#               self.box.send_keys(Keys.BACK_SPACE)
               self.box.clear()
#               self.box.send_keys(Keys.BACK_SPACE)

   def send_term(self, term):
       self.box.send_keys(term.lower())
       self.box.send_keys(Keys.SPACE)  
       # @wait for autocomplete
#       time.sleep(2)
       els = self.ff.find_elements_by_class_name('sbqs_c')
#       for elem in els:
#           print(term, elem.text, "*")        

   def ask(self,term):
#       term = raw_input("Enter Term or enter q to quit: ")
#       for var in term:  
#           if term is not None and term is not 'q':
#               self.send_term(var)
       self.send_letters(term) # @Run through alphabet
       self.box.clear()
    #           term = raw_input("Enter Term or enter q to quit: ") 
       self.ff.quit()
           
#   def close(self,term):
#       if term
          


if __name__ == "__main__":
    r = Belieber()
    a = time.time()
    r.ask('agreement')
    print (time.time()-a)

   
   
   