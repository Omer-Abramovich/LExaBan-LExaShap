SELECT 
	author.name 
FROM 
	author, organization, writes, publication, conference, domain_conference, domain 
WHERE
	author.oid = organization.oid
	AND author.aid = writes.aid
    AND writes.pid = publication.pid
    AND publication.cid = conference.cid
    AND conference.cid = domain_conference.cid
    AND domain_conference.did = domain.did
    AND organization.name = 'Tel Aviv University'
    AND publication.year > 2010
    AND domain.name = 'Databases'
GROUP BY author.name 