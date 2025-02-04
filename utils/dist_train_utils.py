from comm.comm_utils import *

def distributed_train_foo_iter(args, pipeline, device, train_data_loader):
    pp_rank = get_pipeline_parallel_rank()
    first_node =False
    device_gpu = get_device_gpu()
    if  device_gpu==1:
        if pp_rank ==0:
            first_node = True
            print("-----------A100:----------global ID:"+str(args.rank))
    else:
        if pp_rank<=2:
            first_node = True
            print("-----------T4:----------global ID:"+str(args.rank))

    if first_node:
        print(f"        \033[34;40;7m[Python: Rank-id:{args.rank} cuda-id:{args.cuda_id}]\033[0m", end=" ")
        print("Pipeline_rank() == 0")
        total_time = 0
        for i, data in enumerate(train_data_loader):
            input_ids = data['text'].to(device)
            current_iter_time = pipeline.sgd_iter(input_ids, None)
            if i > 0:
                total_time += current_iter_time
            if i >= args.num_iters-1:
                break
        averaged_time = total_time / (args.num_iters - 1)
        print("Finished running ", args.num_iters,
              " iterations, averaged (exclude the first iter) run time:", averaged_time)


    elif get_pipeline_parallel_rank() == get_pipeline_parallel_world_size() - 1 or get_pipeline_parallel_rank()>=11:
        print(f"        \033[34;40;7m[Python: Rank-id:{args.rank} cuda-id:{args.cuda_id}]\033[0m", end=" ")
        print("Pipeline_rank() == size-1")
        for i, data in enumerate(train_data_loader):
            if args.task == 'SeqClassification':
                labels = data['label'].to(device)
            elif args.task == 'Seq2SeqClassification':
                labels = data['text'].to(device)
            else:
                print("Not supported task!")
                assert False
            pipeline.sgd_iter(None, labels)
            if i >= args.num_iters-1:
                break
    else:
        i = 0
        while True:
            pipeline.sgd_iter(None, None)
            i += 1
            if i >= args.num_iters:
                break
