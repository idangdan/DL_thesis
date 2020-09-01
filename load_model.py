model_path = base_dir
netS.load_state_dict(torch.load(model_path + '_S.pt'))
netG.load_state_dict(torch.load(model_path + '_G.pt'))
netD.load_state_dict(torch.load(model_path + '_D.pt'))