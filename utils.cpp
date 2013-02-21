#include "gen-pk.h"
//For CHAR_BIT
#include <limits.h>

/** \file
 * Defines a few small utility functions*/
int print_pk(std::string filename, int nrbins, float * keffs, float * power, int * count)
{
  FILE *fd;
  if(!(fd=fopen(filename.c_str(), "w"))){
     fprintf(stderr,"Error opening file: %s\n",filename.c_str());
     return 0;
  }
  for(int i=0;i<nrbins;i++)
  {
    //if(count[i])
      fprintf(fd,"%e\t%e\t%d\n",keffs[i],power[i],count[i]);
  }
  fclose(fd);
  return nrbins;
}


int print_bi(std::string filename, int nrbins, float * keffs, float * bispec, float* bispeci, int * countbi)
{
  FILE *fd;
  if(!(fd=fopen(filename.c_str(), "w"))){
     fprintf(stderr,"Error opening file: %s\n",filename.c_str());
     return 0;
  }
  fprintf(stderr,"Writing file: %s\n",filename.c_str());
  for(int i=0;i<nrbins;i++) {
    for(int j=0;j<nrbins;j++) {
	for (int k=0;k<nrbins;k++) {
	    	if(countbi[i+nrbins*j+nrbins*nrbins*k])
      			fprintf(fd,"%e\t%e\t%e\t%e\t%e\t%d\n",keffs[i],keffs[j],keffs[k],bispec[i+nrbins*j+nrbins*nrbins*k],bispeci[i+nrbins*j+nrbins*nrbins*k],countbi[i+nrbins*j+nrbins*nrbins*k]);
	}
    }
  }
  fclose(fd);
  return nrbins;
}


/*Returns the maximum value of an array of size size*/
/*int maxarr(int *arr, int size)
{
   int max=*arr;
   while(arr<arr+size)
   {
      max=(max > *(++arr) ? max : *arr);
   }
   return max;
}*/

/*Returns the next power of two. Stolen from wikipedia.*/
int nexttwo(int n)
{
    unsigned int i;
    n--;
    for(i=1;i<sizeof(int)*CHAR_BIT; i<<=1)
       n |= n>>i;
    return ++n; 
}

void help()
{
           fprintf(stderr, "Usage: ./gen-pk -i filenames -o outdir\n"
                           "Outputs one file per particle type, with the name PK-$TYPE-$INPUT\n"
                           "Each output file has three columns, for each bin, k_eff, P(k) and N_modes\n");
           return;
}

std::string type_str(int type)
{
        switch(type)
        {
                case BARYON_TYPE:
                        return "by";
                case DM_TYPE:
                        return "DM";
                case NEUTRINO_TYPE:
                        return "nu";
                default:
                        return "xx";
        }
}
