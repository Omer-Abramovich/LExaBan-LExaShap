SELECT 
	conference.name 
FROM 
	author, organization, writes, publication, conference 
WHERE
	author.oid = organization.oid
	AND author.aid = writes.aid
    AND writes.pid = publication.pid
    AND publication.cid = conference.cid
    AND organization.name = 'University of California San Diego'
    AND publication.year > 2010
GROUP BY conference.name 