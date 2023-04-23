import torch
import os
from os.path import exists
from model import SupConLoss
import subprocess

class Trainer:
    
    def __init__(self, model_name, model, criterion, optimizer, scheduler, epochs, train_loader, val_loader, device, checkpoint_dir, mode, early_stopping, log_periodicity, **kwargs):
        self.model_name = model_name
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.early_stopping = early_stopping
        self.checkpoint_dir = checkpoint_dir
        if self.checkpoint_dir[-1] != '/':
            self.checkpoint_dir += '/'
        self.final_dir = '/scratch/ut_ckp/final/'
        os.makedirs(self.final_dir, exist_ok=True)
        self.mode = mode
        self.log_periodicity = log_periodicity
        self.kwargs = kwargs
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax0 = torch.nn.Softmax(dim=0)
        self.softmax1 = torch.nn.Softmax(dim=1)
        self.contrastive_loss = SupConLoss()
        self.cont_loss_contri = 0.1
        self.test_loader = self.kwargs["test_loader"]
        

    def train(self):
        print(f"Started Training of {self.model_name}")
        self.model = self.model.to(self.device)
        self.model.train()

        for epoch in range(self.epochs):
            running_loss = 0.0

            # Train for an epoch
            for i, (ids, masks, target, var) in enumerate(self.train_loader):
                ids = ids.to(self.device)
                masks = masks.to(self.device)
                # print(ids.shape, masks.shape, target.shape, var)

                if self.mode == 'slot':
                    slots = target.to(self.device)
                    slots = torch.transpose(slots, 0, 1) #  L,N
                    length_sum = var
                else:
                    labels = target.to(self.device)
                    scenarios = var.to(self.device)

                # ids,masks are  B*seq_len
                # labels  B* (#classes)
                loss = 0

                if self.mode == 'slot':
                    vocab_probs = self.model(ids, masks) # L, N, slot_size (157,32,58)
                    # print(len(vocab_probs))
                    # print(vocab_probs[0].shape)
                    seq_len = len(slots)
                    # print(seq_len)
                    for j in range(seq_len):
                        probs = vocab_probs[j]
                        loss += self.criterion(probs, slots[j])
                    loss /= length_sum
            
                else:
                    # out_lbl = self.model(ids, masks) # B * (#classes)
                    out_lbl, out_scn = self.model(ids, masks) # B * (#classes)
                    loss = self.criterion(out_lbl, labels)
                    loss += self.cont_loss_contri * self.contrastive_loss(out_scn, labels)
                
                # print(loss)
                # exit(0)
                # Update the weights
                self.optimizer.zero_grad()
                running_loss += loss
                loss.backward()
                self.optimizer.step()

                if i % self.log_periodicity == (self.log_periodicity - 1):
                    print('[%d, %d] loss: %.6f' %
                          (epoch + 1, i + 1, running_loss / self.log_periodicity), flush=True)
                    running_loss = 0.0

            # Calculate validation loss
            if self.mode=='slot':
                val_loss, accuracy, f1 = self.calculate_slot_loss_accuracy(loader=self.val_loader)
                print('Epoch: %d Validation loss: %.5f accuracy: %.5f F1-score: %.5f' % (epoch + 1, val_loss, accuracy, f1), flush=True)

                test_loss, accuracy, f1 = self.calculate_slot_loss_accuracy(loader=self.test_loader)
                print('Epoch: %d Test loss: %.5f accuracy: %.5f F1-score: %.5f' % (epoch + 1, test_loss, accuracy, f1), flush=True)               
            
            else:
                val_loss, accuracy, accuracy_scn = self.calculate_intent_loss_accuracy(loader=self.val_loader)
                print('Epoch: %d Validation loss: %.5f accuracy: %.5f accuracy_scn: %.5f' % (epoch + 1, val_loss, accuracy, accuracy_scn), flush=True)

                test_loss, accuracy, accuracy_scn = self.calculate_intent_loss_accuracy(loader=self.test_loader)
                print('Epoch: %d Test loss: %.5f accuracy: %.5f accuracy_scn: %.5f' % (epoch + 1, test_loss, accuracy, accuracy_scn), flush=True)

            # Take a scheduler step
            self.scheduler.step(val_loss)

            # Take a early stopping step
            self.early_stopping.step(val_loss)
            
            # Save a checkpoint according to strategy
            self.checkpoint(epoch)

            # Check early stopping to finish training
            if self.early_stopping.stop_training:
                print("Early Stopping the training")
                break
                
            # check external instructions to stop training
            if exists("STOP"):
                print("External instruction to stop the training")
                break

        print('Finished Training')
        self.save_model(self.model_name + "final")

    def calculate_slot_loss_accuracy(self, loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        ne = 0
        tns = [0]*self.kwargs['vocab_size']
        fps = [0]*self.kwargs['vocab_size']
        fns = [0]*self.kwargs['vocab_size']
        tps = [0]*self.kwargs['vocab_size']

        # Test validation data
        with torch.no_grad():
            for i, (ids, mask, slots, length_sum) in enumerate(self.train_loader):
                ids = ids.to(self.device)     # shape N * seq_len(L) * E
                mask = mask.to(self.device)     # shape N * seq_len(L)
                slots = slots.to(self.device) # shape N * seq_len(L)

                slots = torch.transpose(slots, 0, 1)

                # shape seq_len(L) * N * vocab
                vocab_probs = self.model(ids, mask)
                seq_len = len(slots)

                loss = 0
                for j in range(seq_len):
                    slot_probs = vocab_probs[j] # shape N * vocab
                    idxs = slots[j]      # shape N
                    loss += self.criterion(vocab_probs[j], slots[j])

                    # Applying Softmax
                    slot_probs = self.softmax1(slot_probs)

                    # Accuracy
                    for prob, idx in zip(slot_probs, idxs):
                        if torch.argmax(prob) == idx:
                            correct += 1
                            if idx != self.kwargs["pad_index"]:
                                ne += 1
                        total += 1
                    # print(f'{ne}_{correct}__{total}', end=' ')

                    # F1-score
                    for prob, idx in zip(slot_probs, idxs):
                        pred_idx = torch.argmax(prob)
                        if pred_idx == idx:
                            tps[pred_idx] += 1
                        else:
                            fns[idx] += 1
                            fps[pred_idx] += 1
         
                # print()
                # print(f'{ne}_{correct}__{total}')
                loss /= length_sum
                total_loss += loss.data
                
        f1 = 0
        sz = 0
        for i in range(self.kwargs['vocab_size']):
            tns[i] = total - (tps[i]+fps[i]+fns[i])
        for tp, fp, fn, tn in zip(tps, fps, fns, tns):
            # print(f'{tp}__{fp}__{fn}__{tn}', end=' ')
            try:
                precision = tp/(tp+fp)
                recall = tp/(tp+fn)
                f1 += 2 * (precision * recall) / (precision + recall)
                sz += 1
            except:
                continue
        f1 /= sz                
        
        self.model.train()
        return total_loss / len(loader), correct / total, f1

    def calculate_intent_loss_accuracy(self, loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        correct_scn = 0
        total = 0

        # Test validation data
        with torch.no_grad():
            for i, (ids, masks, labels, scenarios) in enumerate(loader):
                ids = ids.to(self.device)
                masks = masks.to(self.device)
                labels = labels.to(self.device)
                scenarios = scenarios.to(self.device)

                # ids,masks are  B*seq_len
                # labels  B* (#classes)
                loss = 0
                # pred_classes = self.model(ids, masks) # B * (#classes)
                pred_classes, pred_scn = self.model(ids, masks) # B * (#classes)
                loss = self.criterion(pred_classes, labels)
                loss += self.cont_loss_contri * self.contrastive_loss(pred_scn, labels)

                # Applying sigmoid
                pred_classes = self.softmax1(pred_classes)
                # pred_scn = self.softmax1(pred_scn)

                # Accumulating loss, accuracy
                total_loss += loss.data
                correct += sum(torch.argmax(pred_classes, dim=1) == labels)
                # correct_scn += sum(torch.argmax(pred_scn, dim=1) == scenarios)
                total += len(labels)


        self.model.train()
        return total_loss / len(loader), correct / total, 0# correct_scn / total

    def save_model(self, file_name):

        if 'final' in file_name:
            print(f'Saving Final Model in {self.final_dir + file_name}')
            torch.save(self.model.state_dict(), f"{self.final_dir + file_name}.pt")
            # # Upload final model in OneDrive
            # cmd = f'rclone copy {self.final_dir + file_name}.pt ut1dr:IS/{self.mode}/'
            # process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
            # output, error = process.communicate()
            # print(output, error)
            # if error == None:
            #     print('Model uploaded successfully to OneDrive')
        else:
            print(f'Saving Model in {self.checkpoint_dir + file_name}')
            torch.save(self.model.state_dict(), f"{self.checkpoint_dir + file_name}.pt")

    
    def evaluate(self, name, loader):
        if self.mode == 'slot':
            loss, accuracy, f1 = self.calculate_slot_loss_accuracy(loader)
            print(f"{name} loss = {loss} \t {name} accuracy = {accuracy} F1-score = {f1}")
        else:
            loss, accuracy, accuracy_scn = self.calculate_intent_loss_accuracy(loader)
            print(f"{name} loss = {loss} \t {name} accuracy = {accuracy} accuracy_scn = {accuracy_scn}")

    def checkpoint(self, epoch):
        save_checkpoint = False
        checkpoint_name = ""
        if self.kwargs["checkpoint_strategy"] == "periodic" and epoch % self.kwargs["checkpoint_periodicity"] == (self.kwargs["checkpoint_periodicity"] - 1):
            save_checkpoint = True
            checkpoint_name = f"checkpoint_{epoch}"
        elif self.kwargs["checkpoint_strategy"] == "best" and self.early_stopping.current_count == 0:
            save_checkpoint = True
            checkpoint_name = "checkpoint_best"
        elif self.kwargs["checkpoint_strategy"] == "both":
            if self.early_stopping.current_count == 0:
                save_checkpoint = True
                checkpoint_name = "checkpoint_best"
            elif epoch % self.kwargs["checkpoint_periodicity"] == (self.kwargs["checkpoint_periodicity"] - 1):
                save_checkpoint = True
                checkpoint_name = f"checkpoint_{epoch}"
            
        if save_checkpoint:
            self.save_model(self.model_name + checkpoint_name)
        

class EarlyStopping:
    def __init__(self, patience):
        self.patience = patience
        self.min_loss = 1e5
        self.current_count = 0
        self.stop_training = False
    
    def step(self, val_loss):
        if val_loss < self.min_loss:
            self.current_count = 0
            self.min_loss = val_loss
        else:
            self.current_count += 1
        
        if self.current_count >= self.patience:
            self.stop_training = True
