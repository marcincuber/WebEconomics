#!/usr/bin/python

class Auction:
	'This class represents an auction of multiple ad slots to multiple advertisers'
	query = ""
	bids = []

	def __init__(self, term, bids1=[]):
		self.query = term
		
		for b in bids1:
			j=0
			#print len(self.bids)
			while j<len(self.bids) and b.value <self.bids[j].value:
				j+=1
			self.bids.insert(j,b)

	'''
	This method accepts a Vector of slots and fills it with the results
	of a VCG auction. The competition for those slots is specified in the bids Vector.
	@param slots a Vector of Slots, which (on entry) specifies only the clickThruRates
	and (on exit) also specifies the name of the bidder who won that slot,
	the price said bidder must pay,
	and the expected profit for the bidder.  
	'''

	def executeVCG(self,slots):
		# array to store tuples with bids and bidder names
		bids = []

		#loop through bids and add them to bids array
		for i in range(0,len(self.bids)):
			bids.append([float(self.bids[i].value), self.bids[i].name])

		#sort bids array
		bids = sorted(bids)
		#number of bids
		length_bids = len(bids)
		#number of slots
		length = len(slots)

		#new array to store bids which will contain one more bid than number of slots
		new_bids = []
		#when there are more bids than slots, pick highest ones that are needed
		if length < length_bids:
			new_bids = bids[-length-1:]
			new_bids = new_bids[0]
			bids = bids[-length:]
		#when there are more slots than bids, then append empty tuples so we have equal number of slots and bidder
		elif length_bids < length:
			for i in range(0, length-length_bids):
				bids.insert(0,[0,0])

		#main loop assiging values to each slot
		for i in range(0, length):
			try:
				#pick the last slot with smallest cliclThru rate
				slot = slots[length-i-1]

				#deal with last slot seperatly,we may need to consider bid value which does not exist in main bid array
				if i == 0:
					try:
						slot.price = (float(slot.clickThruRate) * new_bids[0])
						slot.profit = (float(slot.clickThruRate) * bids[i][0])-slot.price
						slot.bidder = bids[i][1]
					except:
						slot.price = 0
						slot.profit = (float(slot.clickThruRate) * bids[i][0])-slot.price
						slot.bidder = bids[i][1]
				#deal with rest of the slots and assign values
				else:
					slot.price = (((float(slot.clickThruRate) * bids[i-1][0]) + slots[length-i].price)-(float(slots[length-i].clickThruRate)*bids[i-1][0]))
					slot.bidder = bids[i][1]
					slot.profit = (float(slot.clickThruRate) * bids[i][0])-slot.price
			except:
				pass
