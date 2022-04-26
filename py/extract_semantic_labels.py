import json
    
def read_array_from_textfile(filename):
    with open(filename, 'r') as f:
        return [s.strip() for s in f.readlines()]
        
def read_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def write_to_file(filename, data, header):
    with open(filename,'w+') as f:
        f.write(header)
        f.write(data)

def normalize_labels(labels):
    return [s.split(':')[0].strip().split(',')[0].strip().replace(' ','_') for s in labels]

def split_dr_path(path):
    prefix = '/30T/DR/images/releases/'
    postfix = '.jpg'
    assert path.startswith(prefix)
    path = path[len(prefix):]
    assert path.endswith(postfix)
    path = path[:-len(postfix)]
    parts = path.split('/')
    assert len(parts)==2
    return parts
    
def find_semantic_labels(all_labels, top_25):
    ret = []
    for i in top_25:
        ret.append(all_labels[i])
    return ret

def missing_album_cover_statistics(inputlist_uri):
    albums_no_img = 0
    tracks_no_img = []
    for img in inputlist_uri:
        if len(img['images']) == 0:
            albums_no_img +=1
            tracks_no_img.extend(img["track_ids"])
    print("no artwork:\talbums",albums_no_img, "\ttracks", len(tracks_no_img))
    return tracks_no_img

def main():
    #dr_img_list = read_array_from_textfile(''dralbumcovers/dr_img_list.txt'')
    inputlist_uri = read_json('/home/ek/Documents/Thesis/Spotify/AlbumImages/input_to_image_classifier.json')
    tracks_no_img = missing_album_cover_statistics(inputlist_uri)

    sp_img_list = read_array_from_textfile('spotify/spotify_images_w300.txt') 
    img_net_features = normalize_labels(read_array_from_textfile('spotify/imgnet13k_classes.txt')) #'dralbumcovers/imgnet13k_classes.txt'
    all_top25 = read_json('spotify/top25.json') #'dralbumcovers/top25.json'
    corrupted = read_json('spotify/corrupted_images.json')
    
    i = 0
    count = 0
    output_lines = []
    for img in sp_img_list:
        top25 = all_top25[i]
        labels = find_semantic_labels(img_net_features, top25)
        if img in corrupted:
            count += len(inputlist_uri[i]['track_ids'])
            #corrupted_labels = ','.join(labels)
            #print(img,"\n", corrupted_labels)
            i +=1
            continue # exclude corrupted 
        
        labels_str = ','.join(labels)
        #release_id, img_id = split_dr_path(img)
        #output_lines.append(f'{release_id}\t{img_id}\t{labels_str}')
        sp_uris = inputlist_uri[i]['track_ids']
        for uri in sp_uris:
            if uri in tracks_no_img:
                #print(img,"\n", labels_str)
                continue # exclude album tracks with no album cover art
            output_lines.append(f'{uri}\t{labels_str}')
            #print(uri, labels_str)
        i +=1
    print('corrupted img:\talbums',len(corrupted), '\ttracks', count)
    out = '\n'.join(output_lines)+'\n'
    write_to_file('spotify/sp_semantic_labels.tsv',out, "URI\t[labels]\n") #'dralbumcovers/dr_semantic_labels.tsv'

if __name__ == '__main__':
    main()
