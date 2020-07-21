import datetime
import model
import dataloader
import torch

def loss_spa(I, Y):
    loss = torch.mean()

def training_loop(n_epochs, optimizer, model, loss_fn, train_loader):
    for epoch in range(1, n_epochs + 1):
        loss_train = float(0)
        for imgs in train_loader:
            _,outputs,_ = model(imgs)
            loss = loss_fn(outputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()

        if epoch == 1 or epoch % 10 == 0:
            print('{} Epoch {}, Training loss {}'.format(
                datetime.datetime.now(), epoch, 
                loss_train / len(train_loader)))

if __name__ == "__main__":

    dataset = dataloader.lowlight_loader("/home/amrmustafa/vault/Zero-DCE/Dataset_Part1/[0-9]*/") 
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    DCE_net = model.enhance_net_nopool().cuda()
    
    n_epochs = 10
    batch_size = 8
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(DCE_net.parameters(), lr=learning_rate)

    for epoch in range(n_epochs):
        for batch in train_loader:
            print(batch.shape)
