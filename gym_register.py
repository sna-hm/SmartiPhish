from gym.envs.registration import register

register(
    id='MORAPhishDet-v0',
    entry_point='env:Cyberspace',
    #kwargs={X:None,y:None,batch_size:None,output_shape:None,randomize:False,custom_rewards:None}
)
