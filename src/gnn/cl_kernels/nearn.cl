typedef struct latLong
    {
        float lat;
        float lng;
    } LatLong;

__kernel void nearestNeighbor(__global LatLong *d_locations,
							  __global float *d_distances,
							  const int numRecords,
							  const float lat,
							  const float lng) {
	int globalId = get_global_id(0);

    if (globalId >= numRecords) return;
    
    __global LatLong *latLong = d_locations+globalId;
    __global float *dist=d_distances+globalId;
    
    *dist = (float)sqrt((lat-latLong->lat)*(lat-latLong->lat)+(lng-latLong->lng)*(lng-latLong->lng));

}