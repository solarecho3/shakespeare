ala andrej karpathy's nanogpt

modules/model.py

## development notes
- 25 Jan 23. adam. initializing the repository, architecural design, white paper framing.

	We've chosen the Model-View-Presenter as the base design in order to prepare this library for scaling to new datasets and implementations. The repository will come complete with a well-defined wiki, licensing, and other scale requirements. We want to try and future-proof the design from initialization. This design may change as the advantage becomes apparent.
	
- 7 Feb 23. adam. completing the model, optimizing for scale, hardware constraints.

	We've reached our hardware limit on personal machines without some very clever optimization. At writing there are our hyper parameters:
	
	    - device = 'cuda' if torch.cuda.is_available() else 'cpu'
	    - max_iterations = 20_000
	    - evaluation_interval = 500
	    - block_size = 256 # max context length
	    - batch_size = 64 # independent sequences in parallel
	    - trng_pct = .90
	    - learning_rate = 3e-4
	    - manual_seed = 7561
	    - embedding_table_dims = 32
	    - n_heads = 8
	    - n_layers = 8
	    - dropout = 0.2
	    
    	This was our generation:
	
	07-Feb-23 19:20:10 - INFO - Data decoded using char tokenization: 
	
		yess thou say all, dreavets woult that evar
		Dother?

		PRUMINA:
		It put, this alaster, I am my fow you?

		KING EDWARD IV:
		O matas of her, and Jeceral: you be will,
		Who him ilkli; our his friendmet, theseed, Which is Edcose;
		Instinor so a fit Batily hese; and Reeming hou'd
		Like not the my shallow, Wopphy a me yet
		I gate worry all gor of shalke us.

		Lesk RIVIARD I Sembalin:
		And by Verences, ro but speak her with event.
		Noop, thou, say with but too can I part lives!
		And it the flape then se stripow an...
	
