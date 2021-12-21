import pickle as pkl

data = pkl.load(open('test_set.pkl', 'rb'))


def videos_frames(data_dict):
  videos = data_dict.keys()
  num_frames = 0
  for video in videos:
    df = data_dict[video]
    num_frames+= len(df['video'])
  return len(videos), num_frames
for set_name in data.keys():
  print(set_name)
  num_videos, num_frames = videos_frames(data[set_name]['Test_Set'])
  print(num_videos, num_frames)
 
