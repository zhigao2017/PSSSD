function [paris] = generatepairs(label,batchsize,ratio)
pos=1;
paris=ones(batchsize,3);
num=0;
gallery_num=size(label,1);

	for i = 1:(batchsize/(ratio+1))

		sim_1=unidrnd(gallery_num);
		while(1)
			sim_2=unidrnd(gallery_num);
			if label(sim_1)==label(sim_2)
				break;
			end
		end
		paris(num+1,1)=sim_1;
		paris(num+1,2)=sim_2;
		paris(num+1,3)=1;
		num=num+1;

		for j = 1 : ratio
			dis_1=unidrnd(gallery_num);
			while(1)
				dis_2=unidrnd(gallery_num);
				if label(dis_1)~=label(dis_2)
					break;
				end
			end
			paris(num+1,1)=dis_1;
			paris(num+1,2)=dis_2;
			paris(num+1,3)=0;
			num=num+1;
		end


	end
end