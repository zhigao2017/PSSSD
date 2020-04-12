function CY1 = Compute_SPD(SY1)
	number_sets1=length(SY1);
	for tmpC1=1:number_sets1 
	    
	    Y1=SY1{tmpC1};
	   
	    CY1(:,:,tmpC1)= Y1;
	end
end