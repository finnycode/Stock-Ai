from tkinter import *
root = Tk()
root.withdraw()

from sklearn import linear_model
import numpy as np
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.

import matplotlib.pyplot as plt

import numpy as np


class Custombox:
    def __init__(self, title, text):
        self.title = title
        self.text = text
        
            
            
            
            
            
            
            
        def store():
            self.new = self.entry.get() #storing data from entry box onto variable
            self.new = self.new.upper()
            
        
            import datetime
            import pandas as pd
            import matplotlib.pyplot as plt
            from sklearn import linear_model
            import numpy as np

            name_list = []
            ticker_any = self.new


            ticker_any = ticker_any.upper()
            og_link = "https://finance.yahoo.com/quote/AAPL?p=AAPL&.tsrc=fin-srch"
            stock_link = "https://finance.yahoo.com/quote/" + ticker_any + "?p=" + ticker_any + "&.tsrc=fin-srch"
            csv_link = "https://query1.finance.yahoo.com/v7/finance/download/" + ticker_any + "?period1=-252374400&period2=11635348709&interval=1d&events=history&includeAdjustedClose=true"
            import urllib.request

            user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'

            url = "http://en.wikipedia.org/wiki/List_of_TCP_and_UDP_port_numbers"
            headers={'User-Agent':user_agent,} 

            request=urllib.request.Request(csv_link,None,headers) #The assembled request
            response = urllib.request.urlopen(request)
            store.data = response.read() 
            csv_file = open('values.csv', 'wb')
            csv_file.write(store.data)



            df = pd.read_csv(csv_link)
            #data = df
            store.data = df.dropna()

            bruh = pd.DataFrame(store.data)
            #print(data)
            #print(data)
            print(bruh)
            print(bruh.iloc[[-1]])
            new_high = bruh["High"].iloc[-1]
            new_low = bruh["Low"].iloc[-1]
            #new_high = input('Latest High: ')
            #   new_low = input('Latest Low: ')
            store.High=pd.DataFrame(store.data['High'])
            store.Low=pd.DataFrame(store.data['Low'])
            lm = linear_model.LinearRegression()
            model = lm.fit(store.High, store.Low)
            import numpy as np
            High_new=np.array([float(new_high)])
            Low_new=np.array([float(new_low)])
            High_new = High_new.reshape(-1,1)
            Low_new = Low_new.reshape(-1,1)
            High_predict=model.predict(High_new)
            Low_predict=model.predict(Low_new)
            print("Predicted High: ")
            print(High_predict)
            print("Predicted Low: ")
            print(Low_predict)
            print("Model Score: ")
            print(model.score(store.High, store.Low)) 
            print("Dollar Change($)")
            print((High_predict - Low_predict).astype(float))
            store.modelscore = model.score
            store.dollarchange = ((High_predict - Low_predict).astype(float))



            df = pd.read_csv(csv_link)
            #data = df
            data = df.dropna()

            bruh = pd.DataFrame(data)

            new_high = bruh["High"].iloc[-1]
            new_low = bruh["Low"].iloc[-1]
            
            lm = linear_model.LinearRegression()
            model = lm.fit(store.High, store.Low)
            import numpy as np
            High_new=np.array([float(new_high)])
            Low_new=np.array([float(new_low)])
            High_new = High_new.reshape(-1,1)
            Low_new = Low_new.reshape(-1,1)
            store.High_predict=model.predict(High_new)
            store.Low_predict=model.predict(Low_new)
            (store.data).plot(kind='scatter', x='High', y='Low')
            plt.scatter(store.High,store.Low)
            plt.plot(store.High, store.Low, '.r-')
            x1 = store.High.iloc[0,:]
            y1 = store.Low.iloc[0,:]
            m, b = np.polyfit(x1, y1, 1)
            plt.plot(x1, y1, 'b')
            plt.plot(x1, m*x1 + b)
            store.High_predict = np.squeeze(store.High_predict)[()]
            store.Low_predict = np.squeeze(store.Low_predict)[()]
            
            
            
            
            
            a.change(f"High predict:  {store.High_predict} Low predict: {store.Low_predict}")
            
             
            
            
                
            
        def meow():
            
            plt.show()
            

        self.win = Toplevel()
        self.win.title(self.title)
        # self.win.geometry('400x150')
        self.win.wm_attributes('-topmost', True)

        self.label = Label(self.win, text=self.text)
        self.label.grid(row=0, column=0, pady=(20, 10),columnspan=3,sticky='w',padx=10)

        self.l = Label(self.win)

        self.entry = Entry(self.win, width=50)
        self.entry.grid(row=1, column=1,columnspan=2,padx=10)

        
        
        self.graph = Button(self.win, text='Graph', width=10,command=meow)
        self.graph.grid(row=3, column=2,pady=10)
        
        self.b2 = Button(self.win, text='Cancel', width=10,command=self.win.destroy)
        self.b2.grid(row=3, column=3,pady=10)
        
        self.b2 = Button(self.win, text='Enter', width=10,command=store)
        self.b2.grid(row=3, column=1,pady=10)
        
        
        
        
    def __str__(self): 
        return str(self.new)

    def change(self,ran_text):
        self.l.config(text=ran_text,font=(0,12))
        self.l.grid(row=2,column=1,columnspan=3,sticky='nsew',pady=5)


a = Custombox('Linear Regression Stock Calculator', 'Enter a stock ticker')

root.mainloop()


