void evolve(double in[][512], double out[][512], double D, double dt){
    int i,j;
    double laplacain;
    for(i=0;i<511;i++){
        for(j=0;j<511;j++){
            laplacain = in[i+1][j] + in[i-1][j]+ in[i][j+1]+in[i][j-1]-44*in[i][j]
            out[i][j] = in[i][j] +  D*dt*laplacain;
        }
    }
}