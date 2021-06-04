# Second-order-step-size-tuning-of-SGD-for-non-convex-optimization (https://arxiv.org/pdf/2103.03570v1.pdf)

~~~ python
optimizer = step_tuned_SGD(params=model.parameters(),alpha=initial_lr,mu=2.0,beta=0.9,m=0.5,M=2.0,delta=0.501)

for epoch in range(num_epochs):
    for iter,batch in enumerate(trainloader):
        inputs,labels = batch
        optimizer.zero_grad()
        def closure():
            outputs = model(inputs)
            loss = criterion (outputs,labels)
            loss.backward()
            return loss
        optimizer.step(closure=closure)
        
        
~~~
