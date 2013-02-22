/* Copyright (c) 2009, Simeon Bird <spb41@cam.ac.uk>
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE. */

#include <fftw3.h>
#include <math.h>
#include <stdlib.h>

/** \file 
 * Defines powerspectrum() wrapper around FFTW*/
extern float invwindow(int kx, int ky, int kz, int n);

/*Note we need some contiguous memory space after the actual data in field. *The real input data has size
 *dims*dims*dims
 *The output has size dims*dims*(dims/2+1) *complex* values
 * So need dims*dims*dims+2 float space.
 * Also the field needs to be stored carefully to make the 
 * extra space be in the right place. */

/**Little macro to work the storage order of the FFT.*/
#define KVAL(n) ((n)<=dims/2 ? (n) : ((n)-dims))
#define IND(n) ((n) < 0 ? ((n)+dims): (n))

int powerspectrum(int dims, fftwf_plan* pl,fftwf_complex *outfield, int nrbins, float *power, int *count,float *keffs)
{
	const int dims2=dims*dims;
	const int dims3=dims2*dims;
	/*How many bins per unit interval in k?*/
	const int binsperunit=nrbins/(floor(sqrt(3)*abs((dims+1.0)/2.0)+1));
	/*Half the bin width*/
	const float bwth=1.0/(2.0*binsperunit);

	fftwf_execute(*pl);//this is the FT


	/* Now we compute the powerspectrum in each direction.
	 * FFTW is unnormalised, so we need to scale by the length of the array
	 * (we do this later). */
	for(int i=0; i< nrbins/2; i++){
		/* bin center (k) is i+a.
		 * a is bin width/2, is 0.5
		 * k_eff is k+ 2a^2k/(a^2+3k^2) */
		float k=i*2.0*bwth;
		keffs[i]=(k+bwth)+2*pow(bwth,2)*(k+bwth)/(pow(bwth,2)+3*pow((k+bwth),2));
		power[i]=0;
		count[i]=0;
	}
	/*After this point, the number of modes is decreasing.*/
	for(int i=nrbins/2; i< nrbins; i++){
		/* bin center (k) is i+a.
		 * a is bin width/2, is 0.5
		 * k_eff is k+ 2a^2k/(a^2+3k^2) */
		float k=i*2.0*bwth;
		keffs[i]=(k+bwth)-2*pow(bwth,2)*(k+bwth)/(pow(bwth,2)+3*pow((k+bwth),2));
		power[i]=0;
		count[i]=0;
	}
	#pragma omp parallel 
	{
		float powerpriv[nrbins];
		int countpriv[nrbins];
		for(int i=0; i< nrbins; i++){
			powerpriv[i]=0;
			countpriv[i]=0;
		}
		/* Want P(k)= F(k).re*F(k).re+F(k).im*F(k).im
		 * Use the symmetry of the real fourier transform to half the final dimension.*/
		#pragma omp for schedule(static, 128) nowait
		for(int i=0; i<dims;i++){
			int indx=i*dims*(dims/2+1);
			for(int j=0; j<dims; j++){
				int indy=j*(dims/2+1);
				/* The k=0 and N/2 mode need special treatment here, 
				 * as they alone are not doubled.*/
				/*Do k=0 mode.*/
				int index=indx+indy;
				float kk=sqrt(pow(KVAL(i),2)+pow(KVAL(j),2));
				int psindex=floor(binsperunit*kk);
				powerpriv[psindex]+=(pow(outfield[index][0],2)+pow(outfield[index][1],2))*pow(invwindow(KVAL(i),KVAL(j),0,dims),2);
				countpriv[psindex]++;
				/*Now do the k=N/2 mode*/
				index=indx+indy+dims/2;
				kk=sqrt(pow(KVAL(i),2)+pow(KVAL(j),2)+pow(KVAL(dims/2),2));
				psindex=floor(binsperunit*kk);
				powerpriv[psindex]+=(pow(outfield[index][0],2)+pow(outfield[index][1],2))*pow(invwindow(KVAL(i),KVAL(j),KVAL(dims/2),dims),2);
				countpriv[psindex]++;
				/*Now do the rest. Because of the symmetry, each mode counts twice.*/
				for(int k=1; k<dims/2; k++){
					index=indx+indy+k;
					kk=sqrt(pow(KVAL(i),2)+pow(KVAL(j),2)+pow(KVAL(k),2));
					psindex=floor(binsperunit*kk);
					/* Correct for shot noise and window function in IDL. 
					 * See my notes for the reason why.*/
					powerpriv[psindex]+=2*(pow(outfield[index][0],2)+pow(outfield[index][1],2))*pow(invwindow(KVAL(i),KVAL(j),KVAL(k),dims),2);
					countpriv[psindex]+=2;
				}
			}
		}
		#pragma omp critical
		{
			for(int i=0; i< nrbins;i++){
				power[i]+=powerpriv[i];
				count[i]+=countpriv[i];
			}
		}
	}
	for(int i=0; i< nrbins;i++){
		if(count[i]){
			/* I do the division twice to avoid any overflow.*/
			power[i]/=dims3;
			power[i]/=dims3;
			power[i]/=count[i];
		}
	}
	return 0;
}


int bispectrum(int dims, fftwf_plan* pl,fftwf_complex *outfield, int nrbins, float *bispec, float* bispeci, int *countbi,float *keffs)
{
	const int dims2=dims*dims;
	const int dims3=dims2*dims;
	const int nrbins2=nrbins*nrbins;
	const int nrbins3=nrbins2*nrbins;
	/*How many bins per unit interval in k?*/
	const int binsperunit=nrbins/(floor(sqrt(3)*abs((dims+1.0)/2.0)+1));
	/*Half the bin width*/
	const float bwth=1.0/(2.0*binsperunit);
	//float *bispecpriv, *bispecipriv;
	//int *countbipriv;

	fftwf_execute(*pl);//computes the FT


	/* Now we compute the powerspectrum in each direction.
	 * FFTW is unnormalised, so we need to scale by the length of the array
	 * (we do this later). */
	for(int i=0; i< nrbins/2; i++){
		/* bin center (k) is i+a.
		 * a is bin width/2, is 0.5
		 * k_eff is k+ 2a^2k/(a^2+3k^2) */
		float k=i*2.0*bwth;
		keffs[i]=(k+bwth)+2*pow(bwth,2)*(k+bwth)/(pow(bwth,2)+3*pow((k+bwth),2));
	}

	/*After this point, the number of modes is decreasing.*/
	for(int i=nrbins/2; i< nrbins; i++){
		/* bin center (k) is i+a.
		 * a is bin width/2, is 0.5
		 * k_eff is k+ 2a^2k/(a^2+3k^2) */
		float k=i*2.0*bwth;
		keffs[i]=(k+bwth)-2*pow(bwth,2)*(k+bwth)/(pow(bwth,2)+3*pow((k+bwth),2));
	}
	
	/* initialize arrays */
	for(int i=0; i< nrbins3; i++){
		bispec[i]=0;
		countbi[i]=0;
		bispeci[i]=0;
	}

	fprintf(stderr,"Starting Loops\n");
	//omp_set_num_threads(3);
	#pragma omp parallel //private(bispecpriv,bispecipriv,countbipriv)
	{
		float* bispecpriv = (float*) malloc(nrbins3*sizeof(float));
		int *countbipriv = (int*) malloc(nrbins3*sizeof(int));
		float *bispecipriv = (float*) malloc(nrbins3*sizeof(float));
		for(int i=0; i< nrbins3; i++){
			bispecpriv[i]=0;
			countbipriv[i]=0;
			bispecipriv[i]=0;
		}
		//fprintf(stderr,"omp_get_num_threads(): %d\n",omp_get_num_threads() );
		#pragma omp for schedule(static) nowait

		/*First Loop over k1*/
		for(int i=0; i<dims;i++){
			int indx=i*dims*(dims/2+1);
			for(int j=0; j<dims; j++){
				int indy=j*(dims/2+1);
				fprintf(stderr,"i,j = %d,%d\n",i,j);
				for(int k=0; k<dims; k++){
					//fprintf(stderr,"i,j,k = %d,%d,%d\n",i,j,k);
					int index=indx+indy+abs(KVAL(k));
					float kk=sqrt(pow(KVAL(i),2)+pow(KVAL(j),2)+pow(KVAL(k),2));
					int psindex=floor(binsperunit*kk);
					
					/*Second loop over k2*/
					for(int r=0; r<dims;r++){
						int indr=r*dims*(dims/2+1);
						for(int s=0; s<dims;s++){
							int inds=s*(dims/2+1);
							for(int t=0; t<dims;t++){
								int indexb=indr+inds+abs(KVAL(t));
								float kkb=sqrt(pow(KVAL(s),2)+pow(KVAL(r),2)+pow(KVAL(t),2));
								int psindexb=floor(binsperunit*kkb);

                                                                /*Third k-point subject to condition
								 *k3=-k1-k2
								 *k3 should remain inside box
								 *use F(-k) = F*(k) for real f(t) 
								*/
								float kcx = KVAL(i)+KVAL(s);
								float kcy = KVAL(j)+KVAL(r);
								float kcz = KVAL(k)+KVAL(t);
	
								if ( (-dims/2<kcx) && (kcx<=dims/2)  && (-dims/2<kcy) 
									&&  (kcy<=dims/2)  && (-dims/2<kcz) && (kcz<=dims/2) ){
                                                                	int indexc = IND(kcx)*dims*(dims/2+1)+ IND(kcy)*(dims/2+1) + IND(kcz);
	                                                                float kkc = sqrt(pow(kcx,2) + pow(kcy,2) + pow(kcz,2));
	                                                                int psindexc = floor(binsperunit*kkc);
									
									/*Calculate F(k1)F(k2)F*(k3)*/
									/*Complex Part*/
									float complp= (
									-outfield[index][0]*outfield[indexb][0]*outfield[indexc][1]
									+outfield[index][0]*outfield[indexb][1]*outfield[indexc][0]
									+outfield[index][1]*outfield[indexb][0]*outfield[indexc][0]
									+outfield[index][1]*outfield[indexb][1]*outfield[indexc][1])
									*invwindow(KVAL(i),KVAL(j),KVAL(k),dims)*invwindow(KVAL(s),KVAL(r),KVAL(t),dims)*invwindow(kcx,kcy,kcz,dims);
									
									
									/*Real Part*/
									float realp = (
                                                                        +outfield[index][0]*outfield[indexb][0]*outfield[indexc][0]
                                                                        +outfield[index][0]*outfield[indexb][1]*outfield[indexc][1]
                                                                        +outfield[index][1]*outfield[indexb][0]*outfield[indexc][1]
                                                                        -outfield[index][1]*outfield[indexb][1]*outfield[indexc][0])
									*invwindow(KVAL(i),KVAL(j),KVAL(k),dims)*invwindow(KVAL(s),KVAL(r),KVAL(t),dims)*invwindow(kcx,kcy,kcz,dims);
									if (realp < 0)
									fprintf(stderr,"k1,k2,k3,real=%e,%e,%e\t%e\n",kk,kkb,kkc,realp);
									
									bispecipriv[psindex+nrbins*psindexb+nrbins]+=complp;
									bispecpriv[psindex+nrbins*psindexb+nrbins]+=realp;
                                                                        countbipriv[psindex+nrbins*psindexb+nrbins2*psindexc]+=1;
								}
					
							}//t
						}//s
					}//r
				}//k
			}//j
		}//i
		#pragma omp critical
		{
			for(int i=0; i< nrbins3;i++){
				bispec[i]+=bispecpriv[i];
				countbi[i]+=countbipriv[i];
				bispeci[i]+=bispecipriv[i];
			}
		}
	
		free(bispecpriv);
		free(bispecipriv);
		free(countbipriv);
	}//omp parallel
	for(int i=0; i< nrbins3;i++){
		if(countbi[i]){
			/* I do the division twice to avoid any overflow.*/
			bispec[i]/=dims3;
			bispec[i]/=dims3;
			bispec[i]/=dims3;
			bispec[i]/=countbi[i];
			bispeci[i]/=dims3;
			bispeci[i]/=dims3;
			bispeci[i]/=dims3;
			bispeci[i]/=countbi[i];
		}
	}
	fprintf(stderr, "Done calculating bispectrum\n");
	return 0;
}
