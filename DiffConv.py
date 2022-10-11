import torch
import numpy as np
import torch.nn as nn

class DiffConv(nn.Module):

    def __init__(self):
        super(DiffConv, self).__init__()




    def forward(self, x):
        sizeofin = x.size()
        
        ins = sizeofin[0]
        n = sizeofin[1]
        sx = sizeofin[2]
        sy = sizeofin[3]  

        self.output = torch.zeros(ins,n*5,sx,sy)
        self.signInputs = torch.zeros(ins,n*5,sx,sy)
          

        for i in range(0,ins):

            
            oM = 0
            oN = n
            
            self.output[i,oM:oN,0:sx,0:sy]= x[i].clone()
            
            oM = n
            oN = 2*n

            #area = self.output[i,oM:oN,0:sx-1,0:sy]
            self.output[i,oM:oN,0:sx-1,0:sy] = self.output[i,oM:oN,0:sx-1,0:sy].add(x[i,0:n,0:sx-1,0:sy])
            self.output[i,oM:oN,0:sx-1,0:sy] = self.output[i,oM:oN,0:sx-1,0:sy].add(-x[i,0:n,1:sx,0:sy])

            oM = 2*n
            oN = 3*n

            #area = ptr[oM:oN,0:sx,0:sy-1]
            self.output[i,oM:oN,0:sx,0:sy-1]=self.output[i,oM:oN,0:sx,0:sy-1].add(x[i,0:n,0:sx,0:sy-1])
            self.output[i,oM:oN,0:sx,0:sy-1]=self.output[i,oM:oN,0:sx,0:sy-1].add(-x[i,0:n,0:sx,1:sy])
            
            oM = 3*n
            oN = 4*n

            #area = ptr[oM:oN,0:sx-1,0:sy-1]
            self.output[i,oM:oN,0:sx-1,0:sy-1]=self.output[i,oM:oN,0:sx-1,0:sy-1].add(x[i,0:n,0:sx-1,0:sy-1])
            self.output[i,oM:oN,0:sx-1,0:sy-1]=self.output[i,oM:oN,0:sx-1,0:sy-1].add(-x[i,0:n,1:sx,1:sy])

            oM = 4*n
            oN = 5*n
           
            #area = ptr[oM:oN,0:sx-1,0:sy-1]
            self.output[i,oM:oN,0:sx-1,0:sy-1]=self.output[i,oM:oN,0:sx-1,0:sy-1].add(x[i,0:n,1:sx,0:sy-1])
            self.output[i,oM:oN,0:sx-1,0:sy-1]=self.output[i,oM:oN,0:sx-1,0:sy-1].add(-x[i,0:n,0:sx-1,1:sy])
            

        oM = n
        oN = 5*n

           
        #self.signInputs = self.output.sign()
        #self.signInputs[0:ins,0:n,:,:] = torch.ones(ins,n,sx,sy)
        self.output[0:ins,oM:oN,:,:] = self.output[0:ins,oM:oN].abs()

        #print(self.output)

        return self.output