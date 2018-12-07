import torchvision
import torch

def test_neuralnet_sample(net, test_data, classes):

    ## Test the network on test data
    dataiter = iter(test_data)

    # print ground truth
    images,labels = dataiter.next()
    truth_img = torchvision.utils.make_grid(images)
    truth_str = ' '.join('%5s' % classes[labels[j]] for j in range(4))

    # check network conclusions
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    predicted_str = ' '.join('%5s' % classes[predicted[j]] for j in range(4))

    return (truth_img, truth_str, predicted_str)

def test_neuralnet_all(net, test_data, classes):
    ## Check network performance on whole dataset
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10)) 
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in test_data:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1) 
            c = (predicted == labels).squeeze()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    print('Accuracy of the network on the 10000 test images: %d %%' % ( 100 * correct / total))
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))