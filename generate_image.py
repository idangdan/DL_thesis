images_path = base_dir + 'images/'
#fixed_noise_50 = torch.randn(1, 50).to(device)

with torch.no_grad():
    noisev = autograd.Variable(fixed_noise_50)
    samples = netG(noisev)
    samples = samples.view(-1, CHANNELS, IM_SIZE, IM_SIZE)
    samples = samples.mul(0.5).add(0.5)
    samples = samples.cpu().data.numpy()
    for sample_ind in range(10):
      img.imsave(images_path + str(sample_ind) + '_2.png', np.squeeze(samples[sample_ind]))
      plt.matshow(np.squeeze(samples[sample_ind]))
      plt.show()